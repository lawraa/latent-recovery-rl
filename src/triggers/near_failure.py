"""Near-failure triggers for MetaWorld tasks.

Each task has its own trigger class that captures its specific failure modes.
Shared infrastructure (stall detection, cooldown, fire-rate tracking) lives in
the base class NearFailureTrigger so task triggers can inherit and reuse it.

Trigger classes
---------------
  PickPlaceTrigger  — pick-place-v3: two-phase (approach stall + transport
                      stall) plus drop detection.
  PushTrigger       — push-v3: two-phase (approach stall + transport stall).
                      Phase switch uses hand-to-puck proximity instead of
                      grasp_success (which is always 0 for push).
  DoorOpenTrigger   — door-open-v3: two-phase (approach stall + opening
                      stall). Uses in_place_reward for Phase B because
                      obj_to_target is hardcoded 0 in MetaWorld.
  AssemblyTrigger   — assembly-v3: three-phase (approach stall + transport
                      stall + drop detection). Phase B tracks wrench-to-peg
                      distance from obs[-3:] because obj_to_target is
                      hardcoded 0 in MetaWorld assembly.

Factory
-------
    from src.triggers.near_failure import make_trigger
    trigger = make_trigger(cfg.trigger)   # cfg.trigger.type selects the class

Interface (all trigger classes)
--------------------------------
    trigger.reset()                   # call at episode start
    fired = trigger.update(obs, info) # call after every env.step()
    trigger.reset_stats()             # reset fire-rate counters
    trigger.fire_rate                 # fraction of steps where trigger fired
"""
from collections import deque
from typing import Deque, Optional

import numpy as np
import torch


# Observation layout for Meta-World tasks
_HAND_SLICE = slice(0, 3)   # end-effector xyz
_OBJ_SLICE  = slice(4, 7)   # object xyz


# ---------------------------------------------------------------------------
# Base class — shared state, cooldown machinery, stall detection
# ---------------------------------------------------------------------------

class NearFailureTrigger:
    """Base class for near-failure triggers.

    Holds shared hyperparameters, the approach-stall history window,
    cooldown counter, and fire-rate tracking.  Subclasses implement update().

    Args:
        window          : Steps in the sliding progress window.
        min_progress    : Minimum improvement (metres) over the window to be
                          considered making progress.
        cooldown        : Steps to stay triggered after a condition fires.
        success_threshold: Distance below which the task is nearly done —
                          trigger is suppressed.
        pregrasp_close  : Hand-to-object distance below which the pre-grasp
                          stall check is suppressed (hand is about to contact).
    """

    def __init__(
        self,
        window:             int   = 20,
        min_progress:       float = 0.01,
        cooldown:           int   = 15,
        success_threshold:  float = 0.05,
        pregrasp_close:     float = 0.04,
    ):
        self.window            = window
        self.min_progress      = min_progress
        self.cooldown          = cooldown
        self.success_threshold = success_threshold
        self.pregrasp_close    = pregrasp_close

        self._pregrasp_hist: Deque[float] = deque(maxlen=window)
        self._cooldown_left   = 0
        self._n_fires         = 0
        self._n_steps         = 0

    def update(self, obs: np.ndarray, info: dict) -> bool:
        """Return True if near-failure is detected this step."""
        raise NotImplementedError

    def reset(self):
        """Call at the start of every episode."""
        self._pregrasp_hist.clear()
        self._cooldown_left = 0

    @property
    def fire_rate(self) -> float:
        """Fraction of steps on which the trigger has fired (since last reset_stats)."""
        return self._n_fires / max(self._n_steps, 1)

    def reset_stats(self):
        """Reset fire-rate counters (call e.g. once per eval period)."""
        self._n_fires = 0
        self._n_steps = 0

    # ------------------------------------------------------------------
    # Shared helpers

    def _stalled(self, history: Deque[float]) -> bool:
        """True if the range of values in the window is below min_progress."""
        return (max(history) - min(history)) < self.min_progress

    def _fire(self):
        self._cooldown_left = self.cooldown
        self._n_fires += 1

    def _reset_state(self):
        self._cooldown_left = 0
        self._pregrasp_hist.clear()


# ---------------------------------------------------------------------------
# PickPlaceTrigger — two-phase trigger for pick-place-v3
# ---------------------------------------------------------------------------

class PickPlaceTrigger(NearFailureTrigger):
    """Two-phase, hysteresis-aware near-failure detector for pick-place-v3.

    Phase A — pre-grasp (grasp_success = 0)
      Signal : hand-to-object distance from obs[0:3] and obs[4:7].
      NOTE   : info['near_object'] is always 0.0 in Meta-World v3 pick-place —
               do NOT use it.  Read positions directly from the observation.
      Failure: hand not getting closer to the object.

    Phase B — post-grasp (grasp_success = 1)
      Signal : info['obj_to_target'] (object-to-goal distance).
      Failure: object not moving toward the goal.

    Instant condition — drop detection
      Fires immediately when grasp_success flips 1→0 while still far from goal.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._postgrasp_hist: Deque[float] = deque(maxlen=self.window)
        self._prev_grasp = False

    def update(self, obs: np.ndarray, info: dict) -> bool:
        obj_to_target = float(info.get("obj_to_target", float("inf")))
        grasp         = float(info.get("grasp_success", 0.0)) > 0.5

        hand_pos    = obs[_HAND_SLICE]
        obj_pos     = obs[_OBJ_SLICE]
        hand_to_obj = float(np.linalg.norm(hand_pos - obj_pos))

        self._n_steps += 1

        # ---- Already at goal ----
        if obj_to_target < self.success_threshold:
            self._reset_state()
            return False

        # ---- Condition 3: drop detection (instantaneous) ----
        just_dropped = self._prev_grasp and not grasp
        if just_dropped and obj_to_target > self.success_threshold:
            self._prev_grasp = grasp
            self._postgrasp_hist.clear()   # stale post-grasp history is now invalid
            self._fire()
            return True

        # ---- Phase transition: reset the relevant history window ----
        if grasp != self._prev_grasp:
            self._pregrasp_hist.clear()
            self._postgrasp_hist.clear()
        self._prev_grasp = grasp

        # ---- Serve cooldown ----
        if self._cooldown_left > 0:
            self._cooldown_left -= 1
            return True

        # ---- Condition 1: pre-grasp stall ----
        if not grasp:
            # Suppress if hand is already very close (about to contact object)
            if hand_to_obj > self.pregrasp_close:
                self._pregrasp_hist.append(hand_to_obj)
                if (
                    len(self._pregrasp_hist) == self.window
                    and self._stalled(self._pregrasp_hist)
                ):
                    self._fire()
                    return True

        # ---- Condition 2: post-grasp transport stall ----
        else:
            self._postgrasp_hist.append(obj_to_target)
            if (
                len(self._postgrasp_hist) == self.window
                and self._stalled(self._postgrasp_hist)
            ):
                self._fire()
                return True

        return False

    def reset(self):
        super().reset()
        self._postgrasp_hist.clear()
        self._prev_grasp = False

    def _reset_state(self):
        super()._reset_state()
        self._postgrasp_hist.clear()


# ---------------------------------------------------------------------------
# PushTrigger — two-phase approach + transport stall trigger for push-v3
# ---------------------------------------------------------------------------

class PushTrigger(NearFailureTrigger):
    """Two-phase near-failure detector for push-v3.

    Push-v3 has no grasp — the robot pushes the puck directly.
    Phase switch uses hand-to-puck proximity instead of grasp_success
    (which is always 0 for push tasks).

    Phase A (approach): hand_to_puck > pregrasp_close.
        Tracks hand-to-puck distance; fires if the hand stalls approaching.

    Phase B (transport): hand_to_puck <= pregrasp_close.
        Tracks info['obj_to_target'] (puck-to-goal distance); fires if the
        puck stalls while the hand is in contact range.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._transport_hist: Deque[float] = deque(maxlen=self.window)
        self._in_phase_b: bool = False

    def update(self, obs: np.ndarray, info: dict) -> bool:
        obj_to_target = float(info.get("obj_to_target", float("inf")))

        hand_pos    = obs[_HAND_SLICE]
        obj_pos     = obs[_OBJ_SLICE]
        hand_to_obj = float(np.linalg.norm(hand_pos - obj_pos))

        self._n_steps += 1

        # ---- Already at goal ----
        if obj_to_target < self.success_threshold:
            self._reset_state()
            return False

        # ---- Phase transition: clear stale history ----
        in_phase_b = (hand_to_obj <= self.pregrasp_close)
        if in_phase_b != self._in_phase_b:
            self._pregrasp_hist.clear()
            self._transport_hist.clear()
        self._in_phase_b = in_phase_b

        # ---- Serve cooldown ----
        if self._cooldown_left > 0:
            self._cooldown_left -= 1
            return True

        # ---- Phase B: puck transport stall ----
        if in_phase_b:
            self._transport_hist.append(obj_to_target)
            if (
                len(self._transport_hist) == self.window
                and self._stalled(self._transport_hist)
            ):
                self._fire()
                return True

        # ---- Phase A: approach stall ----
        else:
            self._pregrasp_hist.append(hand_to_obj)
            if (
                len(self._pregrasp_hist) == self.window
                and self._stalled(self._pregrasp_hist)
            ):
                self._fire()
                return True

        return False

    def reset(self):
        super().reset()
        self._transport_hist.clear()
        self._in_phase_b = False

    def _reset_state(self):
        super()._reset_state()
        self._transport_hist.clear()


# ---------------------------------------------------------------------------
# DoorOpenTrigger — two-phase approach + opening stall trigger for door-open-v3
# ---------------------------------------------------------------------------

class DoorOpenTrigger(NearFailureTrigger):
    """Two-phase near-failure detector for door-open-v3.

    Phase A (approach): hand far from handle (hand_to_handle > pregrasp_close).
        Tracks hand-to-handle distance; fires if it stalls (not getting closer).

    Phase B (opening): hand close to handle (hand_to_handle <= pregrasp_close).
        Tracks ``info['in_place_reward']`` (door opening progress, 0-1); fires
        if progress stalls (door stopped opening).

    Note: ``info['obj_to_target']`` is hardcoded to 0 in MetaWorld door-open-v3,
    so success is detected via ``in_place_reward > 1 - success_threshold``.
    pregrasp_close should be 0.12 (the funnel radius in the reward function).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._door_hist: Deque[float] = deque(maxlen=self.window)
        self._in_phase_b: bool = False

    def update(self, obs: np.ndarray, info: dict) -> bool:
        hand_pos        = obs[_HAND_SLICE]
        handle_pos      = obs[_OBJ_SLICE]
        hand_to_handle  = float(np.linalg.norm(hand_pos - handle_pos))
        in_place_reward = float(info.get("in_place_reward", 0.0))

        self._n_steps += 1

        # ---- Already at goal (door fully open) ----
        if in_place_reward > 1.0 - self.success_threshold:
            self._reset_state()
            return False

        # ---- Phase transition: clear stale history ----
        in_phase_b = (hand_to_handle <= self.pregrasp_close)
        if in_phase_b != self._in_phase_b:
            self._pregrasp_hist.clear()
            self._door_hist.clear()
        self._in_phase_b = in_phase_b

        # ---- Serve cooldown ----
        if self._cooldown_left > 0:
            self._cooldown_left -= 1
            return True

        # ---- Phase B: track door opening progress ----
        if in_phase_b:
            self._door_hist.append(in_place_reward)
            if (
                len(self._door_hist) == self.window
                and self._stalled(self._door_hist)
            ):
                self._fire()
                return True

        # ---- Phase A: track approach distance ----
        else:
            self._pregrasp_hist.append(hand_to_handle)
            if (
                len(self._pregrasp_hist) == self.window
                and self._stalled(self._pregrasp_hist)
            ):
                self._fire()
                return True

        return False

    def reset(self):
        super().reset()
        self._door_hist.clear()
        self._in_phase_b = False

    def _reset_state(self):
        super()._reset_state()
        self._door_hist.clear()


# ---------------------------------------------------------------------------
# AssemblyTrigger — three-phase trigger for assembly-v3 (nut-on-peg)
# ---------------------------------------------------------------------------

class AssemblyTrigger(NearFailureTrigger):
    """Near-failure detector for assembly-v3 (ring-nut-on-peg).

    The task: pick up a ring nut (wrench) from the table and lower it onto
    a vertical peg at a target position in 3D space.

    Phase B (transport): grasp_success = True.
        Tracks wrench-to-peg distance computed as ||obs[4:7] - obs[-3:]||.
        Fires if the wrench stalls en route to the peg.

    Instant condition — drop detection:
        Fires immediately when grasp_success flips True→False while the
        wrench is still far from the peg.

    Phase A (post-drop re-approach): grasp_success = False AND the agent has
        grasped at least once this episode (_episode_had_grasp = True).
        Tracks hand-to-wrench distance; fires if the hand stalls re-approaching
        after a drop.

    Why Phase A is conditioned on prior within-episode grasp
    ----------------------------------------------------------
    Assembly-v3 requires 1M+ steps of exploration before the base actor
    achieves its first grasp.  There are two qualitatively different Phase A
    situations:

    1. First approach (no prior grasp this episode):
       The agent is still in the discovery phase.  The Q-function has not
       seen a successful grasp and provides no useful correction signal.
       Firing Phase A here injects noise that disrupts exploration — all 4
       decoupled seeds with unconditional Phase A fail at 0% SR.

    2. Re-approach after a drop (_episode_had_grasp = True):
       The agent demonstrated grasp competence this episode, then dropped
       the wrench.  The Q-function has seen a grasp and can guide re-approach.
       This IS near-failure: the agent had the object and lost it.
       Phase A correction here is both principled and potentially helpful.

    The condition `_episode_had_grasp` cleanly separates these two cases
    without any global counters or buffer introspection.  It also means Phase A
    cannot fire until Phase B or drop detection has fired first — guaranteeing
    the correction always sees at least one post-grasp experience before Phase A
    correction can influence re-approach.

    Note: info['obj_to_target'] is hardcoded to 0 in MetaWorld assembly-v3.
    Success is detected via info['in_place_reward'] > 1 - success_threshold.
    Peg position is read directly from obs[-3:] (last 3 obs dims, confirmed
    from sawyer_assembly_peg_v3.py and the expert policy).
    """

    def __init__(self, **kwargs):
        # Remove phase_a_enabled from kwargs if passed from old configs
        kwargs.pop("phase_a_enabled", None)
        super().__init__(**kwargs)
        self._transport_hist: Deque[float] = deque(maxlen=self.window)
        self._prev_grasp: bool = False
        self._episode_had_grasp: bool = False   # True once grasp_success=True this episode

    def update(self, obs: np.ndarray, info: dict) -> bool:
        hand_pos   = obs[_HAND_SLICE]
        wrench_pos = obs[_OBJ_SLICE]
        peg_pos    = obs[-3:]

        hand_to_wrench  = float(np.linalg.norm(hand_pos - wrench_pos))
        wrench_to_peg   = float(np.linalg.norm(wrench_pos - peg_pos))
        grasp           = float(info.get("grasp_success", 0.0)) > 0.5
        in_place_reward = float(info.get("in_place_reward", 0.0))

        self._n_steps += 1

        # Track whether we have grasped at any point this episode
        if grasp:
            self._episode_had_grasp = True

        # ---- Wrench successfully placed on peg ----
        if in_place_reward > 1.0 - self.success_threshold:
            self._reset_state()
            return False

        # ---- Drop detection (instantaneous) ----
        just_dropped = self._prev_grasp and not grasp
        if just_dropped and wrench_to_peg > self.success_threshold:
            self._prev_grasp = grasp
            self._transport_hist.clear()
            self._fire()
            return True

        # ---- Phase transition: clear stale history ----
        if grasp != self._prev_grasp:
            self._pregrasp_hist.clear()
            self._transport_hist.clear()
        self._prev_grasp = grasp

        # ---- Serve cooldown ----
        if self._cooldown_left > 0:
            self._cooldown_left -= 1
            return True

        # ---- Phase B: transport stall (wrench not approaching peg) ----
        if grasp:
            self._transport_hist.append(wrench_to_peg)
            if (
                len(self._transport_hist) == self.window
                and self._stalled(self._transport_hist)
            ):
                self._fire()
                return True

        # ---- Phase A: post-drop re-approach stall ----
        # Only fires if the agent has already grasped this episode — meaning
        # the Q-function has useful signal and this is a genuine re-approach
        # after failure, not the initial discovery approach.
        elif self._episode_had_grasp:
            if hand_to_wrench > self.pregrasp_close:
                self._pregrasp_hist.append(hand_to_wrench)
                if (
                    len(self._pregrasp_hist) == self.window
                    and self._stalled(self._pregrasp_hist)
                ):
                    self._fire()
                    return True

        return False

    def reset(self):
        super().reset()
        self._transport_hist.clear()
        self._prev_grasp = False
        self._episode_had_grasp = False

    def _reset_state(self):
        super()._reset_state()
        self._transport_hist.clear()


# ---------------------------------------------------------------------------
# SweepIntoTrigger — two-phase approach + transport stall for sweep-into-v3
# ---------------------------------------------------------------------------

class SweepIntoTrigger(PushTrigger):
    """Two-phase near-failure detector for sweep-into-v3.

    sweep-into-v3 has the same structure as push-v3: the robot sweeps an
    object into a goal bin with no grasp.  The observation layout and
    info keys are identical:
      - obs[0:3]  — end-effector xyz
      - obs[4:7]  — object xyz
      - info['obj_to_target'] — Euclidean distance from object to goal (live,
                                not hardcoded — confirmed by inspection)

    Phase A (approach): hand_to_obj > pregrasp_close.
        Tracks hand-to-object distance; fires if the hand stalls approaching.

    Phase B (transport): hand_to_obj <= pregrasp_close.
        Tracks info['obj_to_target']; fires if the object stalls while the
        hand is in contact range.

    PushTrigger implements both phases exactly, so no override is needed.
    """
    pass


# ---------------------------------------------------------------------------
# PushWallTrigger — two-phase trigger for push-wall-v3
# ---------------------------------------------------------------------------

class PushWallTrigger(PushTrigger):
    """Two-phase near-failure detector for push-wall-v3.

    push-wall-v3 requires pushing a puck around a fixed wall obstacle via a
    midpoint at (-0.05, 0.77).  Despite the detour, the plain Euclidean
    info['obj_to_target'] is still the right Phase B signal: the wall's
    x-offset (~0.1 m max) is dominated by the y-travel (~0.25 m), so
    obj_to_target decreases monotonically throughout the detour path.
    Any genuine stall — stuck against the wall, stalled at the midpoint, or
    stalled on the final approach — will show a flat obj_to_target window
    and correctly fire the trigger.

    Observation layout and info keys are identical to push-v3:
      - obs[0:3]  — end-effector xyz
      - obs[4:7]  — object xyz
      - info['obj_to_target'] — live Euclidean distance to goal

    PushTrigger implements both phases exactly, so no override is needed.
    pregrasp_close=0.06 is the same generous contact zone as push-v3.
    """
    pass


# ---------------------------------------------------------------------------
# RandomTrigger — fires at a fixed probability each step (ablation baseline)
# ---------------------------------------------------------------------------

class RandomTrigger(NearFailureTrigger):
    """Fires independently at a fixed probability on every step.

    Controls for trigger *frequency* while removing near-failure *targeting*.
    If the heuristic trigger outperforms the random trigger at the same mean
    rate, near-failure state selection is the mechanism — not just firing
    frequency.

    Set fire_prob to match the observed mean fire_rate of the heuristic
    trigger on the same task (~0.05 for pick-place, push-v3, push-wall-v3).

    No cooldown is used: each step is an independent Bernoulli draw so that
    the only difference from the heuristic trigger is *which* states receive
    correction, not the burst structure or rate.
    """

    def __init__(self, fire_prob: float = 0.05, **kwargs):
        super().__init__(**kwargs)
        self.fire_prob = fire_prob

    def update(self, obs: np.ndarray, info: dict) -> bool:
        self._n_steps += 1
        if np.random.random() < self.fire_prob:
            self._n_fires += 1
            return True
        return False

    def reset(self):
        pass  # no episode state to reset


# ---------------------------------------------------------------------------
# DeltaZTrigger — task-agnostic trigger using correction module confidence
# ---------------------------------------------------------------------------

class DeltaZTrigger:
    """Fires when ||Δz|| exceeds the adaptive P-th percentile of recent norms.

    The correction module is trained with L2 regularisation (λ‖Δz‖²), so a
    large ‖Δz‖ can only emerge when the Q-benefit of correction strongly
    outweighs the regularisation penalty.  States with large ‖Δz‖ are exactly
    the states where the correction is "confident" it can improve the action.

    Threshold: P-th percentile (default 90th) of the last ``window_size``
    observed ‖Δz‖ norms collected during training rollout.  The threshold is
    self-calibrating — as the correction module grows during training, the
    threshold rises proportionally, so the trigger always fires on the top
    (100-P)% of states by correction confidence.

    Shared threshold state
    ----------------------
    The computed threshold is written into ``agent._dz_threshold`` so that
    fresh evaluation trigger instances (which start with an empty local
    history) can read the value that training has converged on.  The agent
    must be a ``SACAgentDecoupled`` instance that exposes ``_dz_threshold``
    and ``_dz_threshold_ready`` attributes (added in ``sac_decoupled.py``).

    Interface
    ---------
    Same as heuristic triggers: ``update(obs, info)``, ``reset()``,
    ``reset_stats()``, ``fire_rate``.  Does NOT inherit NearFailureTrigger
    because it requires the agent and has no task-specific geometry.

    Args:
        agent        : SACAgentDecoupled — must expose .actor, .correction,
                       .device, ._dz_threshold, ._dz_threshold_ready
        percentile   : Threshold percentile (90 → fire on top 10% of states
                       by ‖Δz‖).
        window_size  : Rolling window length for percentile computation.
        min_samples  : Minimum observations before the trigger can fire.
                       Prevents premature firing when Δz values have not yet
                       accumulated a representative distribution.
        cooldown     : Steps to remain active after a fire event.  Matches
                       heuristic trigger default of 15.
        update_freq  : Recompute percentile threshold every N steps.
                       np.percentile on 10k floats takes ~0.1 ms, so 100
                       steps means negligible overhead (~1 µs/step average).
    """

    def __init__(
        self,
        agent,
        percentile:  int = 90,
        window_size: int = 10_000,
        min_samples: int = 500,
        cooldown:    int = 15,
        update_freq: int = 100,
    ):
        self._agent        = agent
        self._percentile   = percentile
        self._window_size  = window_size
        self._min_samples  = min_samples
        self._cooldown_v   = cooldown      # stored as _cooldown_v to avoid
        self._update_freq  = update_freq   # shadowing NearFailureTrigger names

        self._dz_history: Deque[float] = deque(maxlen=window_size)
        self._cooldown_left    = 0
        self._n_fires          = 0
        self._n_steps          = 0
        self._steps_since_upd  = 0

    def update(self, obs: np.ndarray, info: dict) -> bool:
        """Return True if the correction should fire on this step.

        Speculatively computes Δz for ``obs``, updates the rolling history,
        periodically refreshes the percentile threshold, then decides whether
        to fire.
        """
        self._n_steps += 1
        self._steps_since_upd += 1

        # ---- Speculatively compute ‖Δz‖ (no gradient, no side-effects) ----
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self._agent.device)
        with torch.no_grad():
            z       = self._agent.actor.encoder(obs_t)
            dz      = self._agent.correction(obs_t, z)
            dz_norm = float(dz.norm().item())

        self._dz_history.append(dz_norm)

        # ---- Periodically recompute percentile and write to agent ----
        if self._steps_since_upd >= self._update_freq:
            self._steps_since_upd = 0
            if len(self._dz_history) >= self._min_samples:
                self._agent._dz_threshold       = float(
                    np.percentile(self._dz_history, self._percentile)
                )
                self._agent._dz_threshold_ready = True

        # ---- Serve active cooldown ----
        if self._cooldown_left > 0:
            self._cooldown_left -= 1
            return True

        # ---- Threshold not yet computed — do not fire ----
        if not self._agent._dz_threshold_ready:
            return False

        # ---- Fire if correction is confident on this state ----
        if dz_norm > self._agent._dz_threshold:
            self._cooldown_left = self._cooldown_v
            self._n_fires += 1
            return True

        return False

    def reset(self):
        """Reset cooldown at episode start.

        The rolling history and shared threshold are intentionally preserved
        across episodes — they represent global training statistics, not
        per-episode state.
        """
        self._cooldown_left = 0

    @property
    def fire_rate(self) -> float:
        """Fraction of steps on which the trigger fired (since last reset_stats)."""
        return self._n_fires / max(self._n_steps, 1)

    def reset_stats(self):
        """Reset fire-rate counters (call once per episode, like heuristic triggers)."""
        self._n_fires = 0
        self._n_steps = 0


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_TRIGGER_REGISTRY = {
    "pick_place":  PickPlaceTrigger,
    "push":        PushTrigger,
    "door_open":   DoorOpenTrigger,
    "assembly":    AssemblyTrigger,
    "sweep_into":  SweepIntoTrigger,
    "push_wall":   PushWallTrigger,
    "random":      RandomTrigger,
}


def make_trigger(cfg, agent: Optional[object] = None):
    """Instantiate the right trigger for a task.

    Reads ``cfg.type`` to select the trigger class.  Defaults to
    ``'pick_place'`` when the field is absent so existing configs that
    pre-date this refactor continue to work unchanged.

    Args:
        cfg  : Namespace with trigger hyperparameters and a ``type`` field.
        agent: SACAgentDecoupled instance.  Required only for
               ``type='delta_z'``; ignored for all heuristic trigger types.

    Returns:
        A trigger instance exposing ``update``, ``reset``, ``reset_stats``,
        and ``fire_rate``.
    """
    type_ = getattr(cfg, "type", "pick_place")

    # DeltaZTrigger has a completely different constructor — handle separately.
    if type_ == "delta_z":
        if agent is None:
            raise ValueError(
                "DeltaZTrigger (type='delta_z') requires the agent to be "
                "passed: make_trigger(cfg, agent=agent)."
            )
        return DeltaZTrigger(
            agent       = agent,
            percentile  = getattr(cfg, "percentile",  90),
            window_size = getattr(cfg, "window_size", 10_000),
            min_samples = getattr(cfg, "min_samples", 500),
            cooldown    = getattr(cfg, "cooldown",    15),
            update_freq = getattr(cfg, "update_freq", 100),
        )

    cls = _TRIGGER_REGISTRY.get(type_)
    if cls is None:
        raise ValueError(
            f"Unknown trigger type {type_!r}. "
            f"Available: {list(_TRIGGER_REGISTRY) + ['delta_z']}"
        )
    kwargs = dict(
        window            = cfg.window,
        min_progress      = cfg.min_progress,
        cooldown          = cfg.cooldown,
        success_threshold = cfg.success_threshold,
        pregrasp_close    = cfg.pregrasp_close,
    )
    if type_ == "random":
        kwargs["fire_prob"] = getattr(cfg, "fire_prob", 0.05)
    return cls(**kwargs)
