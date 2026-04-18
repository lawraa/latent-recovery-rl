"""Soft Actor-Critic (SAC) with automatic entropy tuning.

Design note: the GaussianActor separates its forward pass into two stages:
    1. trunk(obs) → latent z            (exposed for Phase 2 correction)
    2. heads(z)   → (mean, log_std)

The actor's ``forward`` method accepts an optional ``z_override`` argument so
the latent correction module (Phase 2) can inject a corrected latent without
modifying this file.

References:
    Haarnoja et al., "Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL
    with a Stochastic Actor" (2018).
"""
import math
import os
import copy
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# Clamp range for log_std to prevent numerical issues
LOG_STD_MIN = -5
LOG_STD_MAX = 2


# ---------------------------------------------------------------------------
# Network building blocks
# ---------------------------------------------------------------------------

def _build_mlp(in_dim: int, hidden_dim: int, out_dim: int, n_layers: int) -> nn.Sequential:
    """Build a ReLU MLP with n_layers hidden layers."""
    assert n_layers >= 1
    layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
    for _ in range(n_layers - 1):
        layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
    layers.append(nn.Linear(hidden_dim, out_dim))
    return nn.Sequential(*layers)


class GaussianActor(nn.Module):
    """Squashed-Gaussian stochastic actor.

    Architecture:
        trunk  : obs_dim → hidden_dim (the latent z)
        heads  : hidden_dim → action_dim  (mean + log_std)

    The trunk output is the latent representation z that Phase 2 will correct.
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256, n_layers: int = 2):
        super().__init__()

        # Trunk: obs → latent z
        trunk_layers: list[nn.Module] = [nn.Linear(obs_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Tanh()]
        for _ in range(n_layers - 1):
            trunk_layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        self.trunk = nn.Sequential(*trunk_layers)
        self.latent_dim = hidden_dim

        # Action distribution heads
        self.mean_head    = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

        # Small init on output layers → stable early entropy
        for head in (self.mean_head, self.log_std_head):
            nn.init.orthogonal_(head.weight, gain=0.01)
            nn.init.zeros_(head.bias)

    def get_latent(self, obs: torch.Tensor) -> torch.Tensor:
        """Return the trunk latent z (no grad tracking — use for external inspection)."""
        return self.trunk(obs)

    def forward(
        self,
        obs: torch.Tensor,
        z_override: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample an action from the squashed-Gaussian policy.

        Args:
            obs:        (B, obs_dim)
            z_override: if provided, skip the trunk and use this latent instead.
                        This is the injection point for the latent correction module.

        Returns:
            action:   (B, action_dim)  in [-1, 1]
            log_prob: (B, 1)           log π(a|s)
            z:        (B, hidden_dim)  trunk latent (or z_override)
        """
        z = self.trunk(obs) if z_override is None else z_override

        mean    = self.mean_head(z)
        log_std = self.log_std_head(z).clamp(LOG_STD_MIN, LOG_STD_MAX)
        std     = log_std.exp()

        # Reparameterisation trick
        dist  = torch.distributions.Normal(mean, std)
        x_t   = dist.rsample()
        action = torch.tanh(x_t)

        # Log-probability with tanh change-of-variables correction
        log_prob = dist.log_prob(x_t) - torch.log(1.0 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob, z

    @torch.no_grad()
    def get_action(self, obs: torch.Tensor, deterministic: bool = False) -> np.ndarray:
        """Select an action for environment interaction (no gradient)."""
        if deterministic:
            z    = self.trunk(obs)
            mean = self.mean_head(z)
            return torch.tanh(mean).cpu().numpy()
        action, _, _ = self.forward(obs)
        return action.cpu().numpy()


class DoubleCritic(nn.Module):
    """Twin Q-networks for the double-Q trick."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256, n_layers: int = 2):
        super().__init__()
        self.Q1 = _build_mlp(obs_dim + action_dim, hidden_dim, 1, n_layers)
        self.Q2 = _build_mlp(obs_dim + action_dim, hidden_dim, 1, n_layers)

    def forward(
        self, obs: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([obs, action], dim=-1)
        return self.Q1(x), self.Q2(x)


# ---------------------------------------------------------------------------
# SAC Agent
# ---------------------------------------------------------------------------

class SACAgent:
    """Soft Actor-Critic with automatic entropy tuning.

    The actor's latent representation is accessible via ``agent.actor.get_latent(obs)``
    or via the ``z`` returned by ``agent.actor.forward(obs)``.  Phase 2 will
    subclass / wrap this agent to inject the correction module.
    """

    def __init__(self, obs_dim: int, action_dim: int, cfg):
        device_str = getattr(cfg, "device", "cpu")
        self.device  = torch.device(device_str)
        self.gamma   = cfg.gamma
        self.tau     = cfg.tau
        self.auto_alpha  = cfg.auto_alpha
        self._action_dim = action_dim

        # Networks
        self.actor = GaussianActor(
            obs_dim, action_dim, cfg.hidden_dim, cfg.n_layers
        ).to(self.device)

        self.critic = DoubleCritic(
            obs_dim, action_dim, cfg.hidden_dim, cfg.n_layers
        ).to(self.device)

        self.critic_target = copy.deepcopy(self.critic).to(self.device)
        for p in self.critic_target.parameters():
            p.requires_grad = False

        # Entropy temperature α
        if self.auto_alpha:
            self.target_entropy = -float(action_dim)
            init_alpha = float(getattr(cfg, "alpha", 1.0))
            self.log_alpha = torch.tensor(
                [math.log(init_alpha)], requires_grad=True, device=self.device
            )
            self.alpha = init_alpha
            self._alpha_min = float(getattr(cfg, "alpha_min", 0.0))
            alpha_lr = float(getattr(cfg, "alpha_lr", cfg.lr))
            self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=alpha_lr)
        else:
            self.alpha = float(cfg.alpha)

        # Optimisers
        self.actor_optim  = torch.optim.Adam(self.actor.parameters(),  lr=cfg.lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=cfg.lr)

        self._n_updates = 0

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        obs_t  = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        action = self.actor.get_action(obs_t, deterministic=deterministic)
        return action.squeeze(0)

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self, batch: Dict[str, np.ndarray]) -> Dict[str, float]:
        obs      = torch.FloatTensor(batch["obs"]).to(self.device)
        actions  = torch.FloatTensor(batch["actions"]).to(self.device)
        rewards  = torch.FloatTensor(batch["rewards"]).to(self.device)
        next_obs = torch.FloatTensor(batch["next_obs"]).to(self.device)
        not_done = torch.FloatTensor(batch["not_dones"]).to(self.device)

        # ---- Critic update ----
        with torch.no_grad():
            next_action, next_log_pi, _ = self.actor(next_obs)
            q1_next, q2_next = self.critic_target(next_obs, next_action)
            q_next  = torch.min(q1_next, q2_next) - self.alpha * next_log_pi
            q_target = rewards + self.gamma * not_done * q_next

        q1, q2 = self.critic(obs, actions)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # ---- Actor update ----
        new_action, log_pi, _ = self.actor(obs)
        q1_pi, q2_pi = self.critic(obs, new_action)
        actor_loss = (self.alpha * log_pi - torch.min(q1_pi, q2_pi)).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # ---- Alpha update ----
        metrics: Dict[str, float] = {
            "critic_loss": critic_loss.item(),
            "actor_loss":  actor_loss.item(),
            "alpha":       self.alpha,
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
            "actor":          self.actor.state_dict(),
            "critic":         self.critic.state_dict(),
            "critic_target":  self.critic_target.state_dict(),
            "actor_optim":    self.actor_optim.state_dict(),
            "critic_optim":   self.critic_optim.state_dict(),
            "n_updates":      self._n_updates,
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
        self.actor_optim.load_state_dict(ckpt["actor_optim"])
        self.critic_optim.load_state_dict(ckpt["critic_optim"])
        self._n_updates = ckpt.get("n_updates", 0)
        if self.auto_alpha and "log_alpha" in ckpt:
            self.log_alpha.data.copy_(ckpt["log_alpha"])
            self.alpha = self.log_alpha.exp().item()
            self.alpha_optim.load_state_dict(ckpt["alpha_optim"])
