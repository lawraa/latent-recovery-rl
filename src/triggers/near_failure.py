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
from typing import Deque

import numpy as np


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
    """Three-phase near-failure detector for assembly-v3 (ring-nut-on-peg).

    The task: pick up a ring nut (wrench) from the table and lower it onto
    a vertical peg at a target position in 3D space.

    Phase A (approach): grasp_success = False, hand_to_wrench > pregrasp_close.
        Tracks hand-to-wrench distance; fires if the hand stalls approaching
        the wrench.

    Phase B (transport): grasp_success = True.
        Tracks wrench-to-peg distance computed as ||obs[4:7] - obs[-3:]||.
        Fires if the wrench stalls en route to the peg (not getting closer).

    Instant condition — drop detection:
        Fires immediately when grasp_success flips True→False while the
        wrench is still far from the peg.

    Note: info['obj_to_target'] is hardcoded to 0 in MetaWorld assembly-v3.
    Success is detected via info['in_place_reward'] > 1 - success_threshold.
    Peg position is read directly from obs[-3:] (last 3 obs dimensions,
    confirmed from sawyer_assembly_peg_v3.py and the expert policy).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._transport_hist: Deque[float] = deque(maxlen=self.window)
        self._prev_grasp: bool = False

    def update(self, obs: np.ndarray, info: dict) -> bool:
        hand_pos   = obs[_HAND_SLICE]
        wrench_pos = obs[_OBJ_SLICE]
        peg_pos    = obs[-3:]

        hand_to_wrench  = float(np.linalg.norm(hand_pos - wrench_pos))
        wrench_to_peg   = float(np.linalg.norm(wrench_pos - peg_pos))
        grasp           = float(info.get("grasp_success", 0.0)) > 0.5
        in_place_reward = float(info.get("in_place_reward", 0.0))

        self._n_steps += 1

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

        # ---- Phase A: approach stall (hand not approaching wrench) ----
        else:
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

    def _reset_state(self):
        super()._reset_state()
        self._transport_hist.clear()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_TRIGGER_REGISTRY = {
    "pick_place": PickPlaceTrigger,
    "push":       PushTrigger,
    "door_open":  DoorOpenTrigger,
    "assembly":   AssemblyTrigger,
}


def make_trigger(cfg) -> NearFailureTrigger:
    """Instantiate the right trigger for a task.

    Reads ``cfg.type`` to select the trigger class.  Defaults to
    ``'pick_place'`` when the field is absent so existing configs that
    pre-date this refactor continue to work unchanged.

    Args:
        cfg: Namespace with trigger hyperparameters and an optional ``type``
             field (e.g. ``'pick_place'`` or ``'push'``).

    Returns:
        A NearFailureTrigger subclass instance.
    """
    type_ = getattr(cfg, "type", "pick_place")
    cls   = _TRIGGER_REGISTRY.get(type_)
    if cls is None:
        raise ValueError(
            f"Unknown trigger type {type_!r}. "
            f"Available: {list(_TRIGGER_REGISTRY)}"
        )
    kwargs = dict(
        window            = cfg.window,
        min_progress      = cfg.min_progress,
        cooldown          = cfg.cooldown,
        success_threshold = cfg.success_threshold,
        pregrasp_close    = cfg.pregrasp_close,
    )
    return cls(**kwargs)
