"""Learned trigger modules for latent correction gating.

Two predictors are defined here:

BenefitPredictor  (v1–v3, deprecated)
--------------------------------------
Takes (obs, z) and is trained via BCE on Q-advantage labels.
Kept for checkpoint loading compatibility with old runs.
Label = sigmoid(ΔQ / std(ΔQ)) — degenerate because correction is
trained to maximise Q, so ΔQ > 0 almost always → predictor learns
to always fire. See §45 in progress.md.

StallPredictor  (v4, current)
------------------------------
Takes obs only and is trained via weighted BCE on regression-from-peak
stall labels computed online during rollout:

    stall_t = sigmoid((peak_so_far − mean(rewards[t-W:t])) / τ)
    Guard:  if peak_so_far < 0.5: stall_t = 0.0

Regression-from-peak fires when the agent had achieved meaningful reward
(peak > 0.5) but current reward has fallen back — the signature of grasp
loss, transport stall, or placement failure.  Diagnostic confirmed:
Cohen's d > 2.4 in 10/39 obs dimensions at SR=54%.

The predictor is obs-only (no z) because:
  1. Stall detection is grounded in observable task state, not latent space
  2. z changes continuously during training (distribution shift), making it
     a poor additional input for a predictor trained offline on a replay buffer
  3. Removing z makes the predictor genuinely task-agnostic — same architecture
     applies to any task with dense rewards without actor coupling
"""
import torch
import torch.nn as nn


class BenefitPredictor(nn.Module):
    """(Deprecated — v1–v3) MLP: (obs, z) → gate ∈ (0, 1).

    Kept for backward compatibility when loading old checkpoints.
    Do not use for new runs; use StallPredictor instead.
    """

    def __init__(self, obs_dim: int, latent_dim: int, hidden_dim: int = 64):
        super().__init__()
        in_dim = obs_dim + latent_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.orthogonal_(self.net[-1].weight, gain=0.01)
        self.net[-1].bias.data.fill_(-3.0)

    def forward(self, obs: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(torch.cat([obs, z], dim=-1)))


class StallPredictor(nn.Module):
    """obs → gate ∈ (0, 1).  Trained on regression-from-peak stall labels.

    Replaces BenefitPredictor in v4.  Obs-only input removes the circular
    dependency on the actor encoder (z) and makes the predictor task-agnostic.

    Architecture: 2-layer MLP with 64 hidden units (same width as
    BenefitPredictor; stall detection is a binary-ish task).

    Initial output bias = -3 → gate ≈ sigmoid(-3) ≈ 0.05 at init.
    This matches the decoupled-baseline behaviour before the predictor
    has seen enough stall examples to calibrate.

    Args:
        obs_dim:    Dimension of the environment observation.
        hidden_dim: Hidden layer width.
    """

    def __init__(self, obs_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        # Small nonzero output weight preserves gradient flow through hidden
        # layers while keeping initial gate close to 0.05.
        nn.init.orthogonal_(self.net[-1].weight, gain=0.01)
        self.net[-1].bias.data.fill_(-3.0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Compute stall probability.

        Args:
            obs: (B, obs_dim) — raw environment observation.

        Returns:
            gate: (B, 1) ∈ (0, 1), probability that obs is a stall state.
        """
        return torch.sigmoid(self.net(obs))
