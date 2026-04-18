"""SAC with correction gated by the near-failure trigger.

Training  : Correction module is trained always-on, identical to
            SACAgentWithCorrection. The trigger does not affect gradient updates.

Deployment: select_action accepts a `triggered` flag from the training loop.
            When True  → corrected latent z' = z + Δz  (same as always-on).
            When False → base actor latent z, no correction.

This separation lets us directly ablate trigger vs no-trigger on the SAME
trained weights, which is the cleanest comparison for the paper.
"""
import numpy as np
import torch

from src.agents.sac_with_correction import SACAgentWithCorrection


class SACAgentTriggered(SACAgentWithCorrection):
    """Thin subclass: only select_action is changed."""

    def select_action(
        self,
        obs: np.ndarray,
        deterministic: bool = False,
        triggered: bool = False,
    ) -> np.ndarray:
        """Select an action, optionally applying the latent correction.

        Args:
            obs         : Current observation.
            deterministic: If True, take the mean action (no sampling).
            triggered   : If True, apply the correction module.
                          If False, use the base actor directly.
        """
        if triggered:
            # Delegate to always-on parent: z' = z + f(s, z)
            return super().select_action(obs, deterministic=deterministic)

        # Base policy: no correction
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.actor.get_action(obs_t, deterministic=deterministic)
        return action.squeeze(0)
