"""SAC with sequentially trained latent correction.

Experimental baseline for the paper that tests whether concurrent training
is actually necessary.

Protocol
--------
Phase 1 (external, not this class):
    Train a standard SAC baseline to convergence (2.5M steps).
    Save the final checkpoint.

Phase 2 (this class):
    Load the converged baseline checkpoint.
    Freeze the entire actor (all parameters, requires_grad=False).
    Train ONLY the correction module f_phi on top of the frozen base,
    using the same gradient isolation as SACAgentDecoupled (Step 3).
    Also continue updating the critic so it can evaluate corrected actions.
    Alpha is held fixed at the converged baseline value — with the actor
    frozen there is no entropy signal to tune it against.

Prediction
----------
Sequential correction should underperform concurrent decoupled on seeds
where the base policy's early training (the period before Phase 2) was
the bottleneck.  Specifically:
  - Pick-place s3: baseline reaches 0% (never learns to grasp).  The
    frozen actor has no grasp skill.  The correction module has no
    bottleneck transitions to improve on.  Expected result: ~0%.
  - Pick-place s0-s2: baseline reaches 82-96%.  The actor is already
    competent.  Sequential correction may add marginal improvement but
    cannot provide the early-training Q-signal that concurrent correction
    provides when the policy is first discovering grasping.

Comparison to concurrent decoupled
-----------------------------------
Concurrent decoupled trains both the base actor and the correction from
step 1.  During early training (steps 0-1.5M), the trigger fires heavily
on bottleneck states, and the corrected transitions enter the replay buffer
and give the critic stronger Bellman targets.  The base actor learns from
those improved Q-estimates while it is still developing the grasp skill.
Sequential correction cannot provide this: by the time correction training
starts, the base actor is already converged and the bottleneck states
(pre-grasp stalls) are rare.
"""
import math
import os
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F

from src.agents.sac_decoupled import SACAgentDecoupled


class SACAgentSequentialCorrection(SACAgentDecoupled):
    """Correction-only training on top of a frozen pretrained base actor.

    After calling load_baseline(), the actor is frozen and only the
    correction module and critic are updated during training.
    """

    def load_baseline(self, checkpoint_path: str) -> None:
        """Load a converged SAC baseline checkpoint and freeze the actor.

        Args:
            checkpoint_path: Path to a SACAgent final.pt checkpoint.
                             Must contain keys: actor, critic, critic_target.
                             May optionally contain log_alpha.
        """
        ckpt = torch.load(checkpoint_path, map_location=self.device)

        # Backward compat: remap old "trunk.*" keys to "encoder.*"
        actor_sd = {k.replace("trunk.", "encoder.", 1): v for k, v in ckpt["actor"].items()}
        self.actor.load_state_dict(actor_sd)
        self.critic.load_state_dict(ckpt["critic"])
        self.critic_target.load_state_dict(ckpt["critic_target"])

        # Load the converged alpha so correction loss uses a sensible
        # entropy weight.  Alpha will not be updated further.
        if self.auto_alpha and "log_alpha" in ckpt:
            self.log_alpha.data.copy_(ckpt["log_alpha"])
            self.alpha = self.log_alpha.exp().item()

        # Freeze the entire actor — no parameter of the actor will receive
        # gradients from any loss, including the correction loss.
        for param in self.actor.parameters():
            param.requires_grad_(False)

        print(
            f"[sequential] Loaded baseline from {checkpoint_path}\n"
            f"  Actor frozen. Converged alpha = {self.alpha:.4f}"
        )

    # ------------------------------------------------------------------
    # Update: Steps 1 and 3 only.  Step 2 (actor) and Step 4 (alpha)
    # are skipped — the actor and alpha are fixed at the loaded values.
    # ------------------------------------------------------------------

    def update(self, batch: Dict[str, np.ndarray]) -> Dict[str, float]:
        obs      = torch.FloatTensor(batch["obs"]).to(self.device)
        actions  = torch.FloatTensor(batch["actions"]).to(self.device)
        rewards  = torch.FloatTensor(batch["rewards"]).to(self.device)
        next_obs = torch.FloatTensor(batch["next_obs"]).to(self.device)
        not_done = torch.FloatTensor(batch["not_dones"]).to(self.device)

        # ----------------------------------------------------------
        # 1. Critic update.
        #    Bellman target uses the base policy (actor is frozen, so
        #    all actor calls are effectively inference).
        #    The critic keeps learning so it can evaluate the corrected
        #    actions that now enter the replay buffer.
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
        # 2. SKIPPED — actor is frozen.
        # ----------------------------------------------------------

        # ----------------------------------------------------------
        # 3. Correction update — identical to SACAgentDecoupled Step 3.
        #    Actor params have requires_grad=False, but gradients still
        #    flow THROUGH the actor heads to delta_z and then to the
        #    correction module parameters (PyTorch allows gradient flow
        #    through frozen parameters to upstream tensors that require
        #    grad; only the frozen params themselves won't accumulate).
        # ----------------------------------------------------------
        with torch.no_grad():
            z_base = self.actor.encoder(obs)   # frozen encoder

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
        # 4. SKIPPED — alpha is fixed at the baseline converged value.
        # ----------------------------------------------------------

        # ----------------------------------------------------------
        # 5. Soft target update.
        # ----------------------------------------------------------
        for p, p_t in zip(self.critic.parameters(), self.critic_target.parameters()):
            p_t.data.lerp_(p.data, self.tau)

        self._n_updates += 1
        return {
            "critic_loss":             critic_loss.item(),
            "correction_loss":         correction_loss.item(),
            "alpha":                   self.alpha,
            "correction/delta_z_norm": delta_z.detach().norm(dim=-1).mean().item(),
            "correction/reg_loss":     correction_reg.item(),
        }

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(
            {
                "actor":            self.actor.state_dict(),
                "critic":           self.critic.state_dict(),
                "critic_target":    self.critic_target.state_dict(),
                "correction":       self.correction.state_dict(),
                "critic_optim":     self.critic_optim.state_dict(),
                "correction_optim": self.correction_optim.state_dict(),
                "log_alpha":        self.log_alpha.data,
                "n_updates":        self._n_updates,
            },
            path,
        )
