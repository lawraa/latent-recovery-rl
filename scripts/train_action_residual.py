"""Train SAC + decoupled action-space residual correction on pick-place-v3.

Ablation baseline for the paper: correction operates in pre-tanh action
space (Δa) instead of latent space (Δz).  Training loop is identical to
train_decoupled.py — same trigger, same eval protocol, same logging.

Usage:
    python scripts/train_action_residual.py
    python scripts/train_action_residual.py --seed 1 --run-name sac_pp_action_residual_s1 --wandb
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import time
import yaml

from src.envs.metaworld_wrapper import make_env
from src.agents.sac_action_residual import SACAgentActionResidual
from src.triggers.near_failure import make_trigger
from src.utils.replay_buffer import ReplayBuffer
from src.utils.logger import Logger
from src.utils.misc import set_seed, get_device, dict_to_namespace, namespace_to_dict


def evaluate(agent, env, trigger_cfg):
    """Evaluate on all 50 MT1 task variants with the trigger active.

    Returns (mean_reward, success_rate, mean_trigger_rate).
    """
    total_reward    = 0.0
    total_success   = 0.0
    total_trig_rate = 0.0
    n = env.num_tasks

    for idx in range(n):
        obs, info = env.reset_to_task(idx)
        trigger    = make_trigger(trigger_cfg)
        done       = False
        ep_success = 0.0

        while not done:
            triggered = trigger.update(obs, info)
            action    = agent.select_action(obs, deterministic=True, triggered=triggered)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_success  = max(ep_success, info.get("success", 0.0))
            done        = terminated or truncated
            total_reward += reward

        total_success   += ep_success
        total_trig_rate += trigger.fire_rate

    return total_reward / n, total_success / n, total_trig_rate / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",   type=str, default="configs/pick_place_action_residual.yaml")
    parser.add_argument("--task",     type=str, default=None)
    parser.add_argument("--seed",     type=int, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--wandb",    action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = dict_to_namespace(yaml.safe_load(f))

    if args.task     is not None: cfg.env.task          = args.task
    if args.seed     is not None: cfg.training.seed     = args.seed
    if args.run_name is not None: cfg.logging.run_name  = args.run_name
    if args.wandb:                cfg.logging.use_wandb = True

    set_seed(cfg.training.seed)
    cfg.agent.device     = get_device()
    cfg.agent.correction = cfg.correction
    print(f"Device: {cfg.agent.device}")

    log_dir = os.path.join("experiments", cfg.logging.run_name)
    os.makedirs(log_dir, exist_ok=True)

    logger = Logger(
        log_dir   = log_dir,
        use_wandb = cfg.logging.use_wandb,
        project   = cfg.logging.project,
        run_name  = cfg.logging.run_name,
        config    = namespace_to_dict(cfg),
    )
    logger.save_config(namespace_to_dict(cfg))

    env      = make_env(cfg.env.task, seed=cfg.training.seed)
    eval_env = make_env(cfg.env.task, seed=cfg.training.seed)

    obs_dim    = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    print(f"Task: {cfg.env.task} | obs_dim={obs_dim} | action_dim={action_dim}")

    agent   = SACAgentActionResidual(obs_dim, action_dim, cfg.agent)
    buffer  = ReplayBuffer(obs_dim, action_dim, capacity=cfg.training.buffer_capacity)
    trigger = make_trigger(cfg.trigger)

    obs, info  = env.reset()
    trigger.reset()
    ep_reward  = 0.0
    ep_success = 0.0
    ep_steps   = 0
    ep_num     = 0
    t_start    = time.time()

    for step in range(1, cfg.training.total_steps + 1):

        triggered = trigger.update(obs, info)

        if step <= cfg.training.warmup_steps:
            action    = env.action_space.sample()
            triggered = False
        else:
            action = agent.select_action(obs, deterministic=False, triggered=triggered)

        next_obs, reward, terminated, truncated, info = env.step(action)
        ep_reward  += reward
        ep_success  = max(ep_success, info.get("success", 0.0))
        ep_steps   += 1
        buffer.add(obs, action, reward, next_obs, terminated=terminated and not truncated)
        obs = next_obs

        if terminated or truncated:
            ep_num += 1
            trigger.reset()

            if ep_num % 10 == 0 or step % cfg.logging.log_freq == 0:
                elapsed = time.time() - t_start
                print(
                    f"Step {step:7d} | Ep {ep_num:4d} | "
                    f"R {ep_reward:8.2f} | Succ {ep_success:.0f} | "
                    f"TrigRate {trigger.fire_rate:.3f} | "
                    f"EpLen {ep_steps:4d} | {elapsed:.0f}s"
                )
                logger.log(
                    {
                        "train/episode_reward":    ep_reward,
                        "train/episode_success":   ep_success,
                        "train/episode_length":    ep_steps,
                        "train/trigger_fire_rate": trigger.fire_rate,
                        "train/episodes":          ep_num,
                    },
                    step=step,
                )

            trigger.reset_stats()
            obs, info  = env.reset()
            ep_reward  = 0.0
            ep_success = 0.0
            ep_steps   = 0

        if step > cfg.training.warmup_steps and len(buffer) >= cfg.agent.batch_size:
            for _ in range(cfg.training.updates_per_step):
                batch   = buffer.sample(cfg.agent.batch_size)
                metrics = agent.update(batch)

            if step % cfg.logging.log_freq == 0:
                logger.log(
                    {f"train/{k}": v for k, v in metrics.items()},
                    step=step,
                )

        if step % cfg.training.eval_freq == 0:
            eval_r, eval_succ, eval_trig = evaluate(agent, eval_env, cfg.trigger)
            elapsed = time.time() - t_start
            print(
                f"\n{'='*60}\n"
                f"  EVAL  step={step}  reward={eval_r:.2f}  "
                f"success={eval_succ:.1%}  trig_rate={eval_trig:.2%}  "
                f"time={elapsed:.0f}s\n"
                f"{'='*60}\n"
            )
            logger.log(
                {
                    "eval/mean_reward":   eval_r,
                    "eval/success_rate":  eval_succ,
                    "eval/trigger_rate":  eval_trig,
                },
                step=step,
            )
            agent.save(os.path.join(log_dir, f"ckpt_{step:07d}.pt"))

    agent.save(os.path.join(log_dir, "final.pt"))
    logger.close()
    env.close()
    eval_env.close()
    print("Training complete.")


if __name__ == "__main__":
    main()
