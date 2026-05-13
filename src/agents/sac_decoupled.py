"""SAC with decoupled latent correction.

Three-step update per iteration: (1) critic, (2) base actor via standard SAC,
(3) correction on a frozen encoder (torch.no_grad) so correction gradients
cannot reach actor parameters.
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

        # Shared state for DeltaZTrigger — the trigger stores its percentile
        # threshold here so fresh evaluation trigger instances can read the
        # value computed during training (they have empty local histories).
        self._dz_threshold: float      = 0.0
        self._dz_threshold_ready: bool = False


    def select_action(
        self,
        obs: np.ndarray,
        deterministic: bool = False,
        triggered: bool = False,
    ) -> np.ndarray:
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            z = self.actor.encoder(obs_t)
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


    def update(self, batch: Dict[str, np.ndarray]) -> Dict[str, float]:
        obs      = torch.FloatTensor(batch["obs"]).to(self.device)
        actions  = torch.FloatTensor(batch["actions"]).to(self.device)
        rewards  = torch.FloatTensor(batch["rewards"]).to(self.device)
        next_obs = torch.FloatTensor(batch["next_obs"]).to(self.device)
        not_done = torch.FloatTensor(batch["not_dones"]).to(self.device)

        # --- step 1: critic ---
        with torch.no_grad():
            next_action, next_log_pi, _ = self.actor(next_obs)
            q1_next, q2_next = self.critic_target(next_obs, next_action)
            q_next   = torch.min(q1_next, q2_next) - self.alpha * next_log_pi
            q_target = rewards + self.gamma * not_done * q_next

        q1, q2 = self.critic(obs, actions)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # --- step 2: base actor (standard SAC, no correction in graph) ---
        new_action, log_pi, _ = self.actor(obs)
        q1_pi, q2_pi = self.critic(obs, new_action)
        actor_loss = (self.alpha * log_pi - torch.min(q1_pi, q2_pi)).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # --- step 3: correction (encoder frozen, gradients reach only f_phi) ---
        with torch.no_grad():
            z_base = self.actor.encoder(obs)  # no grad to encoder

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

        # --- step 4: alpha update ---
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

        # --- step 5: soft target update ---
        for p, p_t in zip(self.critic.parameters(), self.critic_target.parameters()):
            p_t.data.lerp_(p.data, self.tau)

        self._n_updates += 1
        return metrics


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
        # Backward compat: remap old "trunk.*" keys to new names
        actor_sd = {k.replace("trunk.", "encoder.", 1): v for k, v in ckpt["actor"].items()}
        corr_sd  = {k.replace("trunk.", "net.", 1): v for k, v in ckpt["correction"].items()}
        self.actor.load_state_dict(actor_sd)
        self.critic.load_state_dict(ckpt["critic"])
        self.critic_target.load_state_dict(ckpt["critic_target"])
        self.correction.load_state_dict(corr_sd)
        self.actor_optim.load_state_dict(ckpt["actor_optim"])
        self.critic_optim.load_state_dict(ckpt["critic_optim"])
        self.correction_optim.load_state_dict(ckpt["correction_optim"])
        self._n_updates = ckpt.get("n_updates", 0)
        if self.auto_alpha and "log_alpha" in ckpt:
            self.log_alpha.data.copy_(ckpt["log_alpha"])
            self.alpha = self.log_alpha.exp().item()
            self.alpha_optim.load_state_dict(ckpt["alpha_optim"])
