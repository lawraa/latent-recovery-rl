"""SAC with decoupled latent correction training.

Problem with the original triggered design (SACAgentTriggered)
--------------------------------------------------------------
The always-on update() trains the actor through the corrected path on every
gradient step.  Over time the base actor co-adapts to rely on the correction
module — it learns to produce good *inputs for the correction* rather than
good actions on its own.  When the correction is not applied (~98% of rollout
steps in the triggered setting), the base actor is weak.  Alpha diverges to
~40 as a symptom.

Fix: decoupled training
-----------------------
1. Critic   — Bellman target uses the BASE policy (no correction).
              The Q-function estimates value for the base policy accurately.

2. Actor    — trained with the standard SAC actor loss, no correction in the
              computation graph.  The base policy is independently optimised
              and works on its own at all times.

3. Correction — trained on a FROZEN base latent (torch.no_grad on the trunk).
               No gradient from the correction loss can reach actor parameters.
               The correction learns to be a pure improvement operator on top
               of an already-good base policy.

Deployment
----------
  triggered=False → base actor only          (step 2 guarantees this is good)
  triggered=True  → base actor + correction  (step 3 guarantees improvement)
"""
import math
import os
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F

from src.agents.sac import SACAgent
from src.modules.latent_correction import LatentCorrectionModule


class SACAgentDecoupled(SACAgent):
    """SAC with a correction module trained independently of the base actor."""

    def __init__(self, obs_dim: int, action_dim: int, cfg):
        super().__init__(obs_dim, action_dim, cfg)

        corr_cfg = cfg.correction
        self.correction = LatentCorrectionModule(
            obs_dim      = obs_dim,
            latent_dim   = self.actor.latent_dim,
            hidden_dim   = corr_cfg.hidden_dim,
            n_layers     = corr_cfg.n_layers,
            output_scale = corr_cfg.output_scale,
        ).to(self.device)

        self.correction_optim     = torch.optim.Adam(
            self.correction.parameters(), lr=corr_cfg.lr
        )
        self.correction_reg_coeff = corr_cfg.reg_coeff

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(
        self,
        obs: np.ndarray,
        deterministic: bool = False,
        triggered: bool = False,
    ) -> np.ndarray:
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            z = self.actor.trunk(obs_t)
            if triggered:
                delta_z = self.correction(obs_t, z)
                z_corr  = z + delta_z
                if deterministic:
                    action = torch.tanh(self.actor.mean_head(z_corr))
                else:
                    action, _, _ = self.actor(obs_t, z_override=z_corr)
                return action.cpu().numpy().squeeze(0)
            else:
                return self.actor.get_action(obs_t, deterministic=deterministic).squeeze(0)

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self, batch: Dict[str, np.ndarray]) -> Dict[str, float]:
        obs      = torch.FloatTensor(batch["obs"]).to(self.device)
        actions  = torch.FloatTensor(batch["actions"]).to(self.device)
        rewards  = torch.FloatTensor(batch["rewards"]).to(self.device)
        next_obs = torch.FloatTensor(batch["next_obs"]).to(self.device)
        not_done = torch.FloatTensor(batch["not_dones"]).to(self.device)

        # ----------------------------------------------------------
        # 1. Critic update — Bellman target uses BASE policy only.
        #    The Q-function accurately estimates value for the base
        #    policy, which is what the actor will be trained against.
        # ----------------------------------------------------------
        with torch.no_grad():
            next_action, next_log_pi, _ = self.actor(next_obs)   # base policy
            q1_next, q2_next = self.critic_target(next_obs, next_action)
            q_next   = torch.min(q1_next, q2_next) - self.alpha * next_log_pi
            q_target = rewards + self.gamma * not_done * q_next

        q1, q2 = self.critic(obs, actions)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # ----------------------------------------------------------
        # 2. Base actor update — standard SAC, NO correction.
        #    Gradient flows only through actor trunk + heads.
        #    The base policy learns to be independently good.
        # ----------------------------------------------------------
        new_action, log_pi, _ = self.actor(obs)
        q1_pi, q2_pi = self.critic(obs, new_action)
        actor_loss = (self.alpha * log_pi - torch.min(q1_pi, q2_pi)).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # ----------------------------------------------------------
        # 3. Correction update — trained on FROZEN base latent.
        #    torch.no_grad() on the trunk means zero gradient can
        #    reach actor parameters through this loss.
        #    The correction learns to be a pure improvement operator.
        # ----------------------------------------------------------
        with torch.no_grad():
            z_base = self.actor.trunk(obs)   # frozen: no grad to actor

        delta_z = self.correction(obs, z_base)
        corrected_action, corrected_log_pi, _ = self.actor(
            obs, z_override=z_base + delta_z
        )
        q1_corr, q2_corr = self.critic(obs, corrected_action)
        correction_reg  = delta_z.pow(2).mean()
        correction_loss = (
            (self.alpha * corrected_log_pi - torch.min(q1_corr, q2_corr)).mean()
            + self.correction_reg_coeff * correction_reg
        )

        self.correction_optim.zero_grad()
        correction_loss.backward()
        self.correction_optim.step()

        # ----------------------------------------------------------
        # 4. Alpha update — based on base actor entropy only.
        # ----------------------------------------------------------
        metrics: Dict[str, float] = {
            "critic_loss":             critic_loss.item(),
            "actor_loss":              actor_loss.item(),
            "correction_loss":         correction_loss.item(),
            "alpha":                   self.alpha,
            "correction/delta_z_norm": delta_z.detach().norm(dim=-1).mean().item(),
            "correction/reg_loss":     correction_reg.item(),
        }

        if self.auto_alpha:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            if self._alpha_min > 0.0:
                with torch.no_grad():
                    self.log_alpha.clamp_(min=math.log(self._alpha_min))
            self.alpha = self.log_alpha.exp().item()
            alpha_objective = (log_pi + self.target_entropy).mean().item()
            metrics["alpha_loss"]      = alpha_loss.item()
            metrics["entropy"]         = -log_pi.mean().item()
            metrics["log_alpha"]       = self.log_alpha.item()
            metrics["log_pi_mean"]     = log_pi.mean().item()
            metrics["alpha_objective"] = alpha_objective

        # ----------------------------------------------------------
        # 5. Soft target update
        # ----------------------------------------------------------
        for p, p_t in zip(self.critic.parameters(), self.critic_target.parameters()):
            p_t.data.lerp_(p.data, self.tau)

        self._n_updates += 1
        return metrics

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        ckpt = {
            "actor":             self.actor.state_dict(),
            "critic":            self.critic.state_dict(),
            "critic_target":     self.critic_target.state_dict(),
            "correction":        self.correction.state_dict(),
            "actor_optim":       self.actor_optim.state_dict(),
            "critic_optim":      self.critic_optim.state_dict(),
            "correction_optim":  self.correction_optim.state_dict(),
            "n_updates":         self._n_updates,
        }
        if self.auto_alpha:
            ckpt["log_alpha"]   = self.log_alpha.data
            ckpt["alpha_optim"] = self.alpha_optim.state_dict()
        torch.save(ckpt, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.critic_target.load_state_dict(ckpt["critic_target"])
        self.correction.load_state_dict(ckpt["correction"])
        self.actor_optim.load_state_dict(ckpt["actor_optim"])
        self.critic_optim.load_state_dict(ckpt["critic_optim"])
        self.correction_optim.load_state_dict(ckpt["correction_optim"])
        self._n_updates = ckpt.get("n_updates", 0)
        if self.auto_alpha and "log_alpha" in ckpt:
            self.log_alpha.data.copy_(ckpt["log_alpha"])
            self.alpha = self.log_alpha.exp().item()
            self.alpha_optim.load_state_dict(ckpt["alpha_optim"])
