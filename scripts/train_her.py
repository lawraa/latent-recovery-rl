"""Train SAC + Hindsight Experience Replay (HER) on a Meta-World MT1 task.

HER (Andrychowicz et al., NeurIPS 2017) seeds the replay buffer with
synthetic "success" transitions by relabeling failed episodes: for each
transition at time t, we sample a future step t' and pretend the goal
was the object's position at t'. This gives the agent dense supervision
even when it never reaches the original goal.

Key differences from train_sac.py
-----------------------------------
  1. Uses HERReplayBuffer instead of ReplayBuffer.
  2. Calls add_to_episode() each step instead of add().
  3. Calls flush_episode() at the end of each episode — this writes both
     the real transitions (with original dense reward) and the HER
     relabeled transitions (with binary success reward) to the buffer.
  4. No trigger, no correction module — plain SAC + HER.

Eval is identical to train_sac.py: deterministic policy on all 50 MT1
variants, using the ORIGINAL goal embeddings (obs[36:39] is real goal).

Usage:
    python scripts/train_her.py \\
        --config configs/pick_place_her.yaml \\
        --seed 0 --run-name sac_pp_her_s0 --wandb

    python scripts/train_her.py \\
        --config configs/push_her.yaml \\
        --seed 0 --run-name sac_push_her_s0 --wandb
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse, time, yaml

from src.envs.metaworld_wrapper import make_env
from src.agents.sac import SACAgent
from src.utils.replay_buffer_her import HERReplayBuffer
from src.utils.logger import Logger
from src.utils.misc import set_seed, get_device, dict_to_namespace, namespace_to_dict


# ---------------------------------------------------------------------------
# Evaluation — identical to train_sac.py
# ---------------------------------------------------------------------------

def evaluate(agent: SACAgent, env) -> tuple:
    """Deterministic eval on all 50 MT1 task variants (real goals, no HER)."""
    total_reward = total_success = 0.0
    for idx in range(env.num_tasks):
        obs, info  = env.reset_to_task(idx)
        done       = False
        ep_success = 0.0
        while not done:
            action = agent.select_action(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward  += reward
            ep_success     = max(ep_success, float(info.get("success", 0.0)))
            done           = terminated or truncated
        total_success += ep_success
    return total_reward / env.num_tasks, total_success / env.num_tasks


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",   type=str, default="configs/pick_place_her.yaml")
    parser.add_argument("--task",     type=str, default=None)
    parser.add_argument("--seed",     type=int, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--wandb",    action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = dict_to_namespace(yaml.safe_load(f))

    if args.task     is not None: cfg.env.task         = args.task
    if args.seed     is not None: cfg.training.seed    = args.seed
    if args.run_name is not None: cfg.logging.run_name = args.run_name
    if args.wandb:                cfg.logging.use_wandb = True

    set_seed(cfg.training.seed)
    cfg.agent.device = get_device()
    print(f"Device: {cfg.agent.device}")

    log_dir = os.path.join("experiments", cfg.logging.run_name)
    os.makedirs(log_dir, exist_ok=True)

    logger = Logger(
        log_dir=log_dir, use_wandb=cfg.logging.use_wandb,
        project=cfg.logging.project, run_name=cfg.logging.run_name,
        config=namespace_to_dict(cfg),
    )
    logger.save_config(namespace_to_dict(cfg))

    env      = make_env(cfg.env.task, seed=cfg.training.seed)
    eval_env = make_env(cfg.env.task, seed=cfg.training.seed)
    obs_dim, action_dim = env.observation_space.shape[0], env.action_space.shape[0]
    print(f"Task: {cfg.env.task} | obs_dim={obs_dim} | action_dim={action_dim}")

    agent = SACAgent(obs_dim, action_dim, cfg.agent)

    # HER replay buffer — same capacity as SAC baseline
    her_ratio         = getattr(cfg.training, "her_ratio", 0.8)
    success_threshold = getattr(cfg.training, "success_threshold", 0.05)
    buffer = HERReplayBuffer(
        obs_dim, action_dim,
        capacity          = cfg.training.buffer_capacity,
        her_ratio         = her_ratio,
        success_threshold = success_threshold,
        # obj_slice and goal_slice use defaults (slice(4,7), slice(36,39))
        # verified for MetaWorld pick-place-v3 and push-v3
    )

    print(f"HER ratio: {her_ratio}  |  success threshold: {success_threshold}m")

    obs, _     = env.reset()
    ep_reward  = ep_success = ep_steps = ep_num = 0
    t_start    = time.time()

    for step in range(1, cfg.training.total_steps + 1):

        # --- Action ---
        if step <= cfg.training.warmup_steps:
            action = env.action_space.sample()
        else:
            action = agent.select_action(obs, deterministic=False)

        next_obs, reward, terminated, truncated, info = env.step(action)
        ep_reward  += reward
        ep_success  = max(ep_success, float(info.get("success", 0.0)))
        ep_steps   += 1

        # Stage transition in episode buffer.
        # NOTE: we always pass the real (dense) reward here; HER creates
        # additional relabeled transitions with binary reward in flush_episode().
        # We do NOT cut bootstrapping with `terminated and not truncated` here —
        # that is handled correctly inside flush_episode() for both real and HER
        # transitions. For real transitions: we store the actual terminated flag.
        buffer.add_to_episode(
            obs, action, reward, next_obs,
            terminated=terminated and not truncated,
        )
        obs  = next_obs
        done = terminated or truncated

        if done:
            # Commit real + HER transitions to the main buffer
            buffer.flush_episode()
            ep_num += 1

            if ep_num % 10 == 0 or step % cfg.logging.log_freq == 0:
                elapsed = time.time() - t_start
                print(
                    f"Step {step:7d} | Ep {ep_num:4d} | "
                    f"R {ep_reward:8.2f} | Succ {ep_success:.0f} | "
                    f"EpLen {ep_steps:4d} | BufSize {len(buffer):7d} | {elapsed:.0f}s"
                )
                logger.log({
                    "train/episode_reward":  ep_reward,
                    "train/episode_success": ep_success,
                    "train/episode_length":  ep_steps,
                    "train/episodes":        ep_num,
                    "train/buffer_size":     len(buffer),
                }, step=step)

            obs, _     = env.reset()
            ep_reward  = ep_success = ep_steps = 0

        # --- Updates ---
        if step > cfg.training.warmup_steps and len(buffer) >= cfg.agent.batch_size:
            for _ in range(cfg.training.updates_per_step):
                metrics = agent.update(buffer.sample(cfg.agent.batch_size))
            if step % cfg.logging.log_freq == 0:
                logger.log({f"train/{k}": v for k, v in metrics.items()}, step=step)

        # --- Eval ---
        if step % cfg.training.eval_freq == 0:
            eval_r, eval_sr = evaluate(agent, eval_env)
            elapsed = time.time() - t_start
            print(
                f"\n{'='*60}\n"
                f"  EVAL  step={step}  reward={eval_r:.2f}  "
                f"success={eval_sr:.1%}  time={elapsed:.0f}s\n"
                f"{'='*60}\n"
            )
            logger.log({"eval/mean_reward": eval_r, "eval/success_rate": eval_sr}, step=step)
            agent.save(os.path.join(log_dir, f"ckpt_{step:07d}.pt"))

    agent.save(os.path.join(log_dir, "final.pt"))
    logger.close()
    env.close(); eval_env.close()
    print("Training complete.")


if __name__ == "__main__":
    main()
