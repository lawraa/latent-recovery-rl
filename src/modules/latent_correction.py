"""Learnable latent correction module: f(s, z) → Δz.

The correction is additive:
    z' = z + f(s, z)

Output layer is initialised near-zero so the module starts as a no-op,
preserving base-policy behaviour at the beginning of training.

Phase 3 will gate this behind a near-failure trigger; for now it is
always-on.
"""
import torch
import torch.nn as nn


class LatentCorrectionModule(nn.Module):
    """MLP that maps (obs, latent_z) → Δz.

    Args:
        obs_dim:      Dimension of the environment observation.
        latent_dim:   Dimension of the actor trunk output (z).
        hidden_dim:   Width of the MLP hidden layers.
        n_layers:     Number of hidden layers.
        output_scale: Multiplier on the output — keeps early corrections
                      small without needing careful weight initialisation.
    """

    def __init__(
        self,
        obs_dim: int,
        latent_dim: int,
        hidden_dim: int = 128,
        n_layers: int = 2,
        output_scale: float = 0.1,
    ):
        super().__init__()
        self.latent_dim   = latent_dim
        self.output_scale = output_scale

        in_dim = obs_dim + latent_dim
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        self.trunk = nn.Sequential(*layers)

        self.out = nn.Linear(hidden_dim, latent_dim)
        # Near-zero init → correction ≈ 0 at training start
        nn.init.orthogonal_(self.out.weight, gain=0.01)
        nn.init.zeros_(self.out.bias)

    def forward(self, obs: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Compute the additive latent correction Δz.

        Args:
            obs: (B, obs_dim)
            z:   (B, latent_dim)  trunk output from the actor

        Returns:
            delta_z: (B, latent_dim)
        """
        x = torch.cat([obs, z], dim=-1)
        return self.out(self.trunk(x)) * self.output_scale
