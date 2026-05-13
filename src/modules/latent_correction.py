"""Latent correction module: f(s, z) → Δz, applied as z' = z + Δz."""
import torch
import torch.nn as nn


class LatentCorrectionModule(nn.Module):

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
        self.net = nn.Sequential(*layers)

        self.out = nn.Linear(hidden_dim, latent_dim)
        # Near-zero init → correction ≈ 0 at training start
        nn.init.orthogonal_(self.out.weight, gain=0.01)
        nn.init.zeros_(self.out.bias)

    def forward(self, obs: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, z], dim=-1)
        return self.out(self.net(x)) * self.output_scale
