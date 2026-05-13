"""Action-space residual correction module: f(s, a) → Δa.

The correction is additive in pre-tanh space:
    x_base = atanh(a)          # invert the squashing
    a'     = tanh(x_base + Δa) # apply residual, re-squash

This keeps a' ∈ (-1, 1) without clipping, and gives the correction
module a full unconstrained range to work in regardless of how saturated
a is.

Output layer is initialised near-zero so the module starts as a no-op,
preserving base-policy behaviour at the beginning of training.
"""
import torch
import torch.nn as nn


class ActionResidualModule(nn.Module):
    """MLP that maps (obs, action) → Δa (pre-tanh residual).

    Args:
        obs_dim:      Dimension of the environment observation.
        action_dim:   Dimension of the action space.
        hidden_dim:   Width of the MLP hidden layers.
        n_layers:     Number of hidden layers.
        output_scale: Multiplier on the output — keeps early corrections
                      small without needing careful weight initialisation.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        n_layers: int = 2,
        output_scale: float = 0.1,
    ):
        super().__init__()
        self.action_dim   = action_dim
        self.output_scale = output_scale

        in_dim = obs_dim + action_dim
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        self.net = nn.Sequential(*layers)

        self.out = nn.Linear(hidden_dim, action_dim)
        # Near-zero init → correction ≈ 0 at training start
        nn.init.orthogonal_(self.out.weight, gain=0.01)
        nn.init.zeros_(self.out.bias)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Compute the additive pre-tanh action residual Δa.

        Args:
            obs:    (B, obs_dim)
            action: (B, action_dim)  squashed base action in (-1, 1)

        Returns:
            delta_a: (B, action_dim)  pre-tanh residual
        """
        x = torch.cat([obs, action], dim=-1)
        return self.out(self.net(x)) * self.output_scale
