"""Meta-World MT1 environment wrapper.

Wraps a single Meta-World task into a standard gymnasium.Env interface.
Handles API differences between Meta-World versions gracefully.

Usage:
    env = make_env("reach-v3", seed=0)
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(action)

The `info` dict always contains:
    success (float): 1.0 if the task was solved this step, else 0.0.

Goal positions are resampled from the MT1 training task distribution on
every reset(), giving the policy a distribution of goals to learn from.
"""
import numpy as np

try:
    import gymnasium as gym
    GYM_BASE = gym.Env
except ImportError:
    import gym
    GYM_BASE = gym.Env

try:
    import metaworld
    METAWORLD_AVAILABLE = True
except ImportError:
    METAWORLD_AVAILABLE = False
    print(
        "[MetaWorld] metaworld not installed. "
        "Install with: pip install git+https://github.com/Farama-Foundation/Metaworld.git@main"
    )


class MetaWorldWrapper(GYM_BASE):
    """Wraps a Meta-World MT1 task into a standard gymnasium.Env.

    Goal and object positions are resampled from the MT1 training task
    distribution on every reset(), so the policy learns to generalise
    across the full goal distribution rather than memorising one config.
    Pass different seeds to get reproducible but distinct goal sequences.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, task_name: str = "reach-v3", seed: int = 0, render_mode=None):
        assert METAWORLD_AVAILABLE, "metaworld is not installed."

        self.task_name   = task_name
        self.render_mode = render_mode
        self._seed       = seed

        # Build MT1 benchmark
        mt1     = metaworld.MT1(task_name, seed=seed)
        env_cls = mt1.train_classes[task_name]

        # Instantiate env — older Meta-World versions don't accept render_mode
        try:
            self._env = env_cls(render_mode=render_mode)
        except TypeError:
            self._env = env_cls()

        # Store task list and a seeded RNG for reproducible goal sampling
        self._train_tasks = mt1.train_tasks
        self._task_rng    = np.random.RandomState(seed)

        # Set an initial task so the env is ready before the first reset()
        self._env.set_task(self._train_tasks[self._task_rng.randint(len(self._train_tasks))])

        self.observation_space = self._env.observation_space
        self.action_space      = self._env.action_space
        self.action_space.seed(seed)   # deterministic sample() across processes
        self._max_episode_steps = getattr(self._env, "max_path_length", 500)

    # ------------------------------------------------------------------
    # gymnasium.Env interface
    # ------------------------------------------------------------------

    @property
    def num_tasks(self) -> int:
        """Number of task variants in the MT1 training set (always 50)."""
        return len(self._train_tasks)

    def reset_to_task(self, idx: int):
        """Reset to a specific task variant by index (0–49).

        Used by evaluation loops that want to sweep all 50 variants exactly once,
        which is the standard Meta-World MT1 evaluation protocol.
        """
        self._env.set_task(self._train_tasks[idx])
        result = self._env.reset()
        if isinstance(result, tuple) and len(result) == 2:
            obs, info = result
        else:
            obs, info = result, {}
        return obs.astype(np.float32), info

    def reset(self, *, seed=None, options=None):
        # Resample goal/object position from the training task distribution
        task = self._train_tasks[self._task_rng.randint(len(self._train_tasks))]
        self._env.set_task(task)

        result = self._env.reset()
        # Handle both old gym API (obs,) and new gymnasium API (obs, info)
        if isinstance(result, tuple) and len(result) == 2:
            obs, info = result
        else:
            obs, info = result, {}
        return obs.astype(np.float32), info

    def step(self, action):
        result = self._env.step(action)
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
        else:
            # Old gym: (obs, reward, done, info)
            obs, reward, done, info = result
            terminated, truncated = done, False

        info["success"] = float(info.get("success", 0.0))
        return obs.astype(np.float32), float(reward), terminated, truncated, info

    def render(self):
        return self._env.render()

    def close(self):
        self._env.close()

    @property
    def unwrapped(self):
        return self._env


def make_env(task_name: str = "reach-v3", seed: int = 0, render_mode=None) -> MetaWorldWrapper:
    """Factory function — the canonical way to create an environment in this project."""
    return MetaWorldWrapper(task_name=task_name, seed=seed, render_mode=render_mode)
