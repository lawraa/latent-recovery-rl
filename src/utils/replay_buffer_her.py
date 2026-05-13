"""Hindsight Experience Replay (HER) replay buffer.

Reference: Andrychowicz et al., "Hindsight Experience Replay", NeurIPS 2017.

Strategy: "future" — for each transition at time t, sample a future step
t' uniformly from [t+1, T-1] within the same episode. The object position
at next_obs[t'] becomes the relabeled goal.

Observation layout (MetaWorld 39-dim, verified empirically):
  obs[4:7]   — object xyz (puck / block / wrench)
  obs[36:39] — goal/target xyz (last 3 dims; confirmed: ||obj-goal|| == info['obj_to_target'])

Reward for relabeled transitions:
  Binary sparse: 1.0 if ||next_obs[obj_slice] - relabeled_goal|| < success_threshold, else 0.0
  This is the standard HER approach. The original dense MetaWorld reward is
  kept for non-relabeled (real) transitions.

Termination for relabeled transitions:
  terminated=True when reward_her=1.0 (goal achieved — do not bootstrap from this state).
  terminated=False otherwise (episode continues toward relabeled goal).

Usage:
    buffer = HERReplayBuffer(obs_dim, action_dim, capacity, her_ratio=0.8)

    # In rollout loop (every step):
    buffer.add_to_episode(obs, action, reward, next_obs, terminated)

    # At episode end (terminated or truncated):
    buffer.flush_episode()   # commits real + relabeled transitions to the main buffer
"""
import numpy as np
from typing import List, Tuple

from src.utils.replay_buffer import ReplayBuffer


class HERReplayBuffer(ReplayBuffer):
    """Replay buffer with Hindsight Experience Replay (future strategy).

    Args:
        obs_dim:            Observation dimension.
        action_dim:         Action dimension.
        capacity:           Maximum number of transitions to store.
        her_ratio:          Fraction of transitions to relabel (default 0.8).
        obj_slice:          Slice of observation containing object xyz.
                            Default slice(4, 7) — verified for MetaWorld MT1.
        goal_slice:         Slice of observation containing goal xyz.
                            Default slice(36, 39) — last 3 dims, verified for MetaWorld MT1.
        success_threshold:  Object-to-goal distance (metres) below which the
                            relabeled task is considered solved.
                            Default 0.05 — matches MetaWorld _TARGET_RADIUS.
    """

    def __init__(
        self,
        obs_dim:            int,
        action_dim:         int,
        capacity:           int   = 1_000_000,
        her_ratio:          float = 0.8,
        obj_slice:          slice = slice(4, 7),
        goal_slice:         slice = slice(36, 39),
        success_threshold:  float = 0.05,
    ):
        super().__init__(obs_dim, action_dim, capacity)
        self.her_ratio          = her_ratio
        self.obj_slice          = obj_slice
        self.goal_slice         = goal_slice
        self.success_threshold  = success_threshold

        # Temporary storage for the current episode.
        # Each element: (obs, action, reward, next_obs, terminated)
        self._episode: List[Tuple] = []

    # ------------------------------------------------------------------
    # Episode interface
    # ------------------------------------------------------------------

    def add_to_episode(
        self,
        obs:        np.ndarray,
        action:     np.ndarray,
        reward:     float,
        next_obs:   np.ndarray,
        terminated: bool,
    ) -> None:
        """Buffer one transition for the current episode.

        Call this every rollout step instead of add().
        Call flush_episode() at episode end.
        """
        self._episode.append((
            obs.copy(), action.copy(), float(reward),
            next_obs.copy(), bool(terminated),
        ))

    def flush_episode(self) -> None:
        """Commit the buffered episode to the main replay buffer.

        For each transition t:
          1. Always add the original (real) transition with the original dense reward.
          2. With probability her_ratio (and only if t < T-1 so a future step exists),
             add one HER transition:
             - Sample future step t' uniformly from [t+1, T-1]
             - achieved_goal = next_obs[t'][obj_slice]  (where the object ended up)
             - Replace goal_slice in obs and next_obs with achieved_goal
             - Recompute reward: 1.0 if object reached achieved_goal at t+1, else 0.0
             - Set terminated = True iff reward_her == 1.0 (goal achieved; stop bootstrap)

        Clears the episode buffer after committing.
        """
        ep  = self._episode
        T   = len(ep)

        for t, (obs, action, reward, next_obs, terminated) in enumerate(ep):

            # ---- 1. Always store the original transition ----
            self.add(obs, action, reward, next_obs, terminated)

            # ---- 2. HER relabeling (future strategy) ----
            # Need at least one future step to sample from.
            if t >= T - 1:
                continue
            if np.random.random() >= self.her_ratio:
                continue

            # Sample a future timestep t' ∈ [t+1, T-1]
            t_future = np.random.randint(t + 1, T)

            # The achieved goal is the object position in the NEXT state at t'
            # (what the object actually moved to after the action at t')
            achieved_goal = ep[t_future][3][self.obj_slice].copy()  # next_obs of t_future

            # Build relabeled observations: swap out the goal component
            obs_her      = obs.copy()
            obs_her[self.goal_slice]      = achieved_goal

            next_obs_her = next_obs.copy()
            next_obs_her[self.goal_slice] = achieved_goal

            # Recompute reward: did the object reach achieved_goal at step t+1?
            dist_at_t1  = float(np.linalg.norm(next_obs[self.obj_slice] - achieved_goal))
            reward_her  = 1.0 if dist_at_t1 < self.success_threshold else 0.0

            # Terminate the HER transition iff the relabeled goal was achieved
            # (do not bootstrap Q from a goal-achieved state)
            terminated_her = (reward_her == 1.0)

            self.add(obs_her, action, reward_her, next_obs_her, terminated_her)

        self._episode = []

    # ------------------------------------------------------------------
    # Sanity helpers
    # ------------------------------------------------------------------

    @property
    def episode_len(self) -> int:
        """Length of the episode currently being collected."""
        return len(self._episode)
