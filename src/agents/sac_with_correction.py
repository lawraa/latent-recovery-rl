"""SAC agent with an always-on latent correction module.

Extends SACAgent by injecting LatentCorrectionModule into the actor's
forward pass via the existing z_override hook.

Key changes vs. base SACAgent
------------------------------
1. select_action:  z' = z + f(s,z) before sampling the action.
2. critic target:  next-state action also uses the corrected policy
                   (keeps the Bellman target consistent with data collection).
3. actor update:   actor loss backpropagates through the correction module;
                   both actor_optim and correction_optim are stepped.
4. regularisation: L2 penalty on ||Δz|| keeps corrections from growing
                   unboundedly (coeff controlled by cfg.correction.reg_coeff).

Logging extras (returned in update metrics dict)
-------------------------------------------------
  correction/delta_z_norm  — mean L2 norm of Δz over the batch
  correction/reg_loss      — regularisation term magnitude
"""
import math
import os
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F

from src.agents.sac import SACAgent
from src.modules.latent_correction import LatentCorrectionModule


class SACAgentWithCorrection(SACAgent):

    def __init__(self, obs_dim: int, action_dim: int, cfg):
        # Build base SAC components (actor, critic, alpha, optimisers)
        super().__init__(obs_dim, action_dim, cfg)

        corr_cfg = cfg.correction
        self.correction = LatentCorrectionModule(
            obs_dim      = obs_dim,
            latent_dim   = self.actor.latent_dim,
            hidden_dim   = corr_cfg.hidden_dim,
            n_layers     = corr_cfg.n_layers,
            output_scale = corr_cfg.output_scale,
        ).to(self.device)

        self.correction_optim = torch.optim.Adam(
            self.correction.parameters(), lr=corr_cfg.lr
        )
        self.correction_reg_coeff = corr_cfg.reg_coeff

    # ------------------------------------------------------------------
    # Internal helper: actor forward with correction applied
    # ------------------------------------------------------------------

    def _corrected_forward(self, obs: torch.Tensor):
        """Run actor forward with latent corrected by f(s, z).

        Returns (action, log_prob, z, delta_z).
        Gradients flow through actor trunk, actor heads, and correction module.
        """
        z       = self.actor.trunk(obs)
        delta_z = self.correction(obs, z)
        action, log_prob, _ = self.actor(obs, z_override=z + delta_z)
        return action, log_prob, z, delta_z

    # ------------------------------------------------------------------
    # Action selection (used during environment interaction)
    # ------------------------------------------------------------------

    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            z       = self.actor.get_latent(obs_t)
            delta_z = self.correction(obs_t, z)
            z_corr  = z + delta_z
            if deterministic:
                mean   = self.actor.mean_head(z_corr)
                action = torch.tanh(mean)
            else:
                action, _, _ = self.actor(obs_t, z_override=z_corr)
        return action.cpu().numpy().squeeze(0)

    # ------------------------------------------------------------------
    # Update (one gradient step)
    # ------------------------------------------------------------------

    def update(self, batch: Dict[str, np.ndarray]) -> Dict[str, float]:
        obs      = torch.FloatTensor(batch["obs"]).to(self.device)
        actions  = torch.FloatTensor(batch["actions"]).to(self.device)
        rewards  = torch.FloatTensor(batch["rewards"]).to(self.device)
        next_obs = torch.FloatTensor(batch["next_obs"]).to(self.device)
        not_done = torch.FloatTensor(batch["not_dones"]).to(self.device)

        # ---- Critic update ----
        # Next-state action must come from the CORRECTED policy so the
        # Bellman target is consistent with how data was collected.
        with torch.no_grad():
            z_next       = self.actor.trunk(next_obs)
            dz_next      = self.correction(next_obs, z_next)
            next_action, next_log_pi, _ = self.actor(
                next_obs, z_override=z_next + dz_next
            )
            q1_next, q2_next = self.critic_target(next_obs, next_action)
            q_next  = torch.min(q1_next, q2_next) - self.alpha * next_log_pi
            q_target = rewards + self.gamma * not_done * q_next

        q1, q2 = self.critic(obs, actions)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # ---- Actor + Correction update ----
        new_action, log_prob, _, delta_z = self._corrected_forward(obs)
        q1_pi, q2_pi = self.critic(obs, new_action)

        correction_reg = delta_z.pow(2).mean()
        actor_loss = (
            (self.alpha * log_prob - torch.min(q1_pi, q2_pi)).mean()
            + self.correction_reg_coeff * correction_reg
        )

        self.actor_optim.zero_grad()
        self.correction_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        self.correction_optim.step()

        # ---- Alpha update ----
        metrics: Dict[str, float] = {
            "critic_loss":           critic_loss.item(),
            "actor_loss":            actor_loss.item(),
            "alpha":                 self.alpha,
            "correction/delta_z_norm": delta_z.detach().norm(dim=-1).mean().item(),
            "correction/reg_loss":   correction_reg.item(),
        }

        if self.auto_alpha:
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            if self._alpha_min > 0.0:
                with torch.no_grad():
                    self.log_alpha.clamp_(min=math.log(self._alpha_min))
            self.alpha = self.log_alpha.exp().item()
            alpha_objective = (log_prob + self.target_entropy).mean().item()
            metrics["alpha_loss"]      = alpha_loss.item()
            metrics["entropy"]         = -log_prob.mean().item()
            metrics["log_alpha"]       = self.log_alpha.item()
            metrics["log_pi_mean"]     = log_prob.mean().item()
            metrics["alpha_objective"] = alpha_objective

        # ---- Soft target update ----
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
