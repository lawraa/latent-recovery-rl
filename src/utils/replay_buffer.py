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
        self.not_dones = np.zeros((capacity, 1),         dtype=np.float32)

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        terminated: bool,
    ):
        """Add a single transition.

        Args:
            terminated: True only for genuine episode endings (task success /
                        failure), NOT for time-limit truncations. SAC should
                        still bootstrap from time-limit-truncated states.
        """
        self.obs[self.ptr]       = obs
        self.actions[self.ptr]   = action
        self.rewards[self.ptr]   = reward
        self.next_obs[self.ptr]  = next_obs
        self.not_dones[self.ptr] = 1.0 - float(terminated)

        self.ptr  = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict:
        idx = np.random.randint(0, self.size, size=batch_size)
        return {
            "obs":       self.obs[idx],
            "actions":   self.actions[idx],
            "rewards":   self.rewards[idx],
            "next_obs":  self.next_obs[idx],
            "not_dones": self.not_dones[idx],
        }

    def __len__(self) -> int:
        return self.size
