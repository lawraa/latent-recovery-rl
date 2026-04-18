"""SAC with decoupled action-space residual correction.

Mirrors SACAgentDecoupled exactly but operates in action space instead of
latent space.  This is the paper's ablation baseline: instead of correcting
the actor's internal latent z, we correct the output action directly.

Correction application
----------------------
    x_base = atanh(base_action)          # invert tanh squashing
    x_corr = x_base + Δa                 # additive residual in pre-tanh space
    a'     = tanh(x_corr)                # re-squash → stays in (-1, 1)

Training (identical decoupling to SACAgentDecoupled)
----------------------------------------------------
1. Critic   — Bellman target uses BASE policy, no correction.
2. Actor    — standard SAC loss, no correction in the graph.
3. Correction — trained on FROZEN base action (torch.no_grad on actor).
               Log π for the corrected action is evaluated under the frozen
               base distribution N(mean_base, std_base).
4. Alpha    — based on base actor entropy only.
"""
import math
import os
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F

from src.agents.sac import SACAgent
from src.agents.sac import LOG_STD_MIN, LOG_STD_MAX
from src.modules.action_residual import ActionResidualModule


class SACAgentActionResidual(SACAgent):
    """SAC with a decoupled action-space residual correction module."""

    def __init__(self, obs_dim: int, action_dim: int, cfg):
        super().__init__(obs_dim, action_dim, cfg)

        corr_cfg = cfg.correction
        self.correction = ActionResidualModule(
            obs_dim      = obs_dim,
            action_dim   = action_dim,
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
            if triggered:
                z = self.actor.trunk(obs_t)
                if deterministic:
                    base_action = torch.tanh(self.actor.mean_head(z))
                else:
                    base_action, _, _ = self.actor(obs_t)
                delta_a    = self.correction(obs_t, base_action).clamp(-2.0, 2.0)
                x_base     = torch.atanh(base_action.clamp(-1 + 1e-6, 1 - 1e-6))
                action     = torch.tanh(x_base + delta_a)
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
        # ----------------------------------------------------------
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

        # ----------------------------------------------------------
        # 2. Base actor update — standard SAC, NO correction.
        # ----------------------------------------------------------
        new_action, log_pi, _ = self.actor(obs)
        q1_pi, q2_pi = self.critic(obs, new_action)
        actor_loss = (self.alpha * log_pi - torch.min(q1_pi, q2_pi)).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # ----------------------------------------------------------
        # 3. Correction update — trained on FROZEN base action.
        #
        # We freeze the entire actor so no gradient reaches actor
        # parameters through this loss.
        #
        # Log π for the corrected action is computed under the frozen
        # base distribution N(mean_base, std_base) evaluated at the
        # corrected pre-tanh value x_corr = atanh(base_action) + Δa.
        # ----------------------------------------------------------
        with torch.no_grad():
            z_base       = self.actor.trunk(obs)
            mean_base    = self.actor.mean_head(z_base)
            log_std_base = self.actor.log_std_head(z_base).clamp(LOG_STD_MIN, LOG_STD_MAX)
            std_base     = log_std_base.exp()
            # Sample from base distribution (reparameterisation)
            eps          = torch.randn_like(mean_base)
            x_t          = mean_base + std_base * eps     # pre-tanh base sample
            base_action  = torch.tanh(x_t)                # squashed base action

        # Correction operates outside no_grad — gradients flow only through delta_a
        delta_a    = self.correction(obs, base_action).clamp(-2.0, 2.0)
        x_corr     = x_t + delta_a                        # corrected pre-tanh value
        corr_action = torch.tanh(x_corr)

        # Log prob of x_corr under frozen N(mean_base, std_base), with tanh Jacobian
        log_prob_gauss = (
            -0.5 * ((x_corr - mean_base) / std_base).pow(2)
            - log_std_base
            - 0.5 * math.log(2 * math.pi)
        )
        corr_log_pi = (
            log_prob_gauss - torch.log(1.0 - corr_action.pow(2) + 1e-6)
        ).sum(dim=-1, keepdim=True)

        q1_corr, q2_corr = self.critic(obs, corr_action)
        correction_reg  = delta_a.pow(2).mean()
        correction_loss = (
            (self.alpha * corr_log_pi - torch.min(q1_corr, q2_corr)).mean()
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
            "correction/delta_a_norm": delta_a.detach().norm(dim=-1).mean().item(),
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
