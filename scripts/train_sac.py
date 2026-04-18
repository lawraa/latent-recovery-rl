"""Train a SAC baseline on a Meta-World MT1 task.

Usage:
    python scripts/train_sac.py                          # uses configs/sac_baseline.yaml
    python scripts/train_sac.py --task push-v3 --seed 1
    python scripts/train_sac.py --wandb --run-name sac_reach_s0
"""
import sys
import os

# Allow running from the repo root without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import time
import yaml

from src.envs.metaworld_wrapper import make_env
from src.agents.sac import SACAgent
from src.utils.replay_buffer import ReplayBuffer
from src.utils.logger import Logger
from src.utils.misc import set_seed, get_device, dict_to_namespace, namespace_to_dict


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def evaluate(agent: SACAgent, env):
    """Evaluate on all 50 MT1 task variants (standard Meta-World protocol).

    Returns (mean_reward, success_rate) where success_rate is the fraction
    of the 50 task variants on which the agent achieved success.
    """
    total_reward  = 0.0
    total_success = 0.0
    n = env.num_tasks
    for idx in range(n):
        obs, info = env.reset_to_task(idx)
        done       = False
        ep_success = 0.0
        while not done:
            action = agent.select_action(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward  += reward
            ep_success     = max(ep_success, info.get("success", 0.0))
            done           = terminated or truncated
        total_success += ep_success
    return total_reward / n, total_success / n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train SAC on Meta-World MT1")
    parser.add_argument("--config",    type=str, default="configs/sac_baseline.yaml")
    parser.add_argument("--task",      type=str, default=None,  help="Override env.task")
    parser.add_argument("--seed",      type=int, default=None,  help="Override training.seed")
    parser.add_argument("--run-name",  type=str, default=None,  help="Override logging.run_name")
    parser.add_argument("--wandb",     action="store_true",     help="Enable wandb logging")
    args = parser.parse_args()

    # ---- Load config ----
    with open(args.config) as f:
        cfg = dict_to_namespace(yaml.safe_load(f))

    # CLI overrides
    if args.task     is not None: cfg.env.task            = args.task
    if args.seed     is not None: cfg.training.seed       = args.seed
    if args.run_name is not None: cfg.logging.run_name    = args.run_name
    if args.wandb:                cfg.logging.use_wandb   = True

    # ---- Seed + device ----
    set_seed(cfg.training.seed)
    device = get_device()
    cfg.agent.device = device
    print(f"Device: {device}")

    # ---- Experiment directory ----
    log_dir = os.path.join("experiments", cfg.logging.run_name)
    os.makedirs(log_dir, exist_ok=True)

    # ---- Logger ----
    logger = Logger(
        log_dir   = log_dir,
        use_wandb = cfg.logging.use_wandb,
        project   = cfg.logging.project,
        run_name  = cfg.logging.run_name,
        config    = namespace_to_dict(cfg),
    )
    logger.save_config(namespace_to_dict(cfg))

    # ---- Environments ----
    # Both envs use the same seed so their goal sequences are reproducible.
    # Goals are resampled on every reset() from the MT1 training distribution,
    # so the policy learns to generalise across goal positions.
    env      = make_env(cfg.env.task, seed=cfg.training.seed)
    eval_env = make_env(cfg.env.task, seed=cfg.training.seed)

    obs_dim    = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    print(f"Task: {cfg.env.task} | obs_dim={obs_dim} | action_dim={action_dim}")

    # ---- Agent + Buffer ----
    agent  = SACAgent(obs_dim, action_dim, cfg.agent)
    buffer = ReplayBuffer(obs_dim, action_dim, capacity=cfg.training.buffer_capacity)

    # ---- Training loop ----
    obs, _        = env.reset()
    ep_reward     = 0.0
    ep_success    = 0.0
    ep_steps      = 0
    ep_num        = 0
    t_start       = time.time()

    for step in range(1, cfg.training.total_steps + 1):

        # --- Interaction ---
        if step <= cfg.training.warmup_steps:
            action = env.action_space.sample()
        else:
            action = agent.select_action(obs, deterministic=False)

        next_obs, reward, terminated, truncated, info = env.step(action)
        ep_reward  += reward
        ep_success  = max(ep_success, info.get("success", 0.0))
        ep_steps   += 1

        # For SAC Bellman targets: only cut bootstrapping at genuine terminations,
        # not at time-limit truncations.
        buffer.add(obs, action, reward, next_obs, terminated=terminated and not truncated)
        obs  = next_obs
        done = terminated or truncated

        if done:
            ep_num += 1
            if ep_num % 10 == 0 or step % cfg.logging.log_freq == 0:
                elapsed = time.time() - t_start
                print(
                    f"Step {step:7d} | Ep {ep_num:4d} | "
                    f"R {ep_reward:8.2f} | Succ {ep_success:.0f} | "
                    f"EpLen {ep_steps:4d} | {elapsed:.0f}s"
                )
                logger.log(
                    {
                        "train/episode_reward":  ep_reward,
                        "train/episode_success": ep_success,
                        "train/episode_length":  ep_steps,
                        "train/episodes":        ep_num,
                    },
                    step=step,
                )
            obs, _     = env.reset()
            ep_reward  = 0.0
            ep_success = 0.0
            ep_steps   = 0

        # --- Updates ---
        if step > cfg.training.warmup_steps and len(buffer) >= cfg.agent.batch_size:
            for _ in range(cfg.training.updates_per_step):
                batch   = buffer.sample(cfg.agent.batch_size)
                metrics = agent.update(batch)

            if step % cfg.logging.log_freq == 0:
                logger.log(
                    {f"train/{k}": v for k, v in metrics.items()},
                    step=step,
                )

        # --- Evaluation ---
        if step % cfg.training.eval_freq == 0:
            eval_reward, eval_success = evaluate(agent, eval_env)
            elapsed = time.time() - t_start
            print(
                f"\n{'='*60}\n"
                f"  EVAL  step={step}  reward={eval_reward:.2f}  "
                f"success={eval_success:.1%}  time={elapsed:.0f}s\n"
                f"{'='*60}\n"
            )
            logger.log(
                {"eval/mean_reward": eval_reward, "eval/success_rate": eval_success},
                step=step,
            )
            ckpt_path = os.path.join(log_dir, f"ckpt_{step:07d}.pt")
            agent.save(ckpt_path)

    # ---- Final save ----
    agent.save(os.path.join(log_dir, "final.pt"))
    logger.close()
    env.close()
    eval_env.close()
    print("Training complete.")


if __name__ == "__main__":
    main()
