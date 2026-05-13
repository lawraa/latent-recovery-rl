"""Simple uniform-random replay buffer backed by pre-allocated numpy arrays."""
import numpy as np


class ReplayBuffer:
    def __init__(self, obs_dim: int, action_dim: int, capacity: int = 1_000_000):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0

        self.obs      = np.zeros((capacity, obs_dim),    dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim),    dtype=np.float32)
        self.actions  = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards  = np.zeros((capacity, 1),          dtype=np.float32)
        # 1 - done: 0 at genuine terminal states so we don't bootstrap over them
        self.not_dones    = np.zeros((capacity, 1), dtype=np.float32)
        # Stall label for v4 learned trigger: regression-from-peak score ∈ [0, 1].
        # 0.0 when gate is not yet active or label is unavailable (v1-v3 compatible).
        self.stall_labels = np.zeros((capacity, 1), dtype=np.float32)

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        terminated: bool,
        stall_label: float = 0.0,
    ):
        """Add a single transition.

        Args:
            terminated:   True only for genuine episode endings (task success /
                          failure), NOT for time-limit truncations.
            stall_label:  Regression-from-peak stall score ∈ [0, 1], computed
                          online during rollout.  Defaults to 0.0 (backward
                          compatible with scripts that don't pass it).
        """
        self.obs[self.ptr]          = obs
        self.actions[self.ptr]      = action
        self.rewards[self.ptr]      = reward
        self.next_obs[self.ptr]     = next_obs
        self.not_dones[self.ptr]    = 1.0 - float(terminated)
        self.stall_labels[self.ptr] = stall_label

        self.ptr  = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict:
        idx = np.random.randint(0, self.size, size=batch_size)
        return {
            "obs":          self.obs[idx],
            "actions":      self.actions[idx],
            "rewards":      self.rewards[idx],
            "next_obs":     self.next_obs[idx],
            "not_dones":    self.not_dones[idx],
            "stall_labels": self.stall_labels[idx],
        }

    def sample_recent(self, batch_size: int, n_recent: int) -> dict:
        """Sample uniformly from the most recent n_recent transitions.

        The buffer is circular so the last n_recent writes end at ptr-1.
        Uses modular arithmetic to handle wrap-around correctly.
        Falls back gracefully if n_recent > size (uses all available data).
        """
        n_available = min(n_recent, self.size)
        # Offsets from ptr: -n_available, ..., -1  →  indices of last n_available writes
        offsets = np.arange(-n_available, 0)
        recent_indices = (self.ptr + offsets) % self.capacity
        replace = batch_size > n_available
        idx = np.random.choice(recent_indices, size=batch_size, replace=replace)
        return {
            "obs":          self.obs[idx],
            "actions":      self.actions[idx],
            "rewards":      self.rewards[idx],
            "next_obs":     self.next_obs[idx],
            "not_dones":    self.not_dones[idx],
            "stall_labels": self.stall_labels[idx],
        }

    def __len__(self) -> int:
        return self.size
