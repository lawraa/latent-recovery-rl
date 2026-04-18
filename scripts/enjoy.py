"""Watch a trained policy act in the environment.

Usage:
    # Live window (requires display)
    python scripts/enjoy.py --run-name sac_reach_s0 --task reach-v3

    # Load a specific checkpoint instead of final.pt
    python scripts/enjoy.py --run-name sac_reach_s0 --ckpt experiments/sac_reach_s0/ckpt_0100000.pt

    # Save video instead of opening a window
    python scripts/enjoy.py --run-name sac_reach_s0 --save-video --n-episodes 5

    # Random policy (sanity check before training is done)
    python scripts/enjoy.py --task reach-v3 --random
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import time
import numpy as np
import torch
import yaml

from src.envs.metaworld_wrapper import make_env
from src.agents.sac import SACAgent
from src.utils.misc import dict_to_namespace, get_device
from src.triggers.near_failure import make_trigger


def needs_vertical_flip(env, camera_name: str) -> bool:
    """Return True if the named camera's image will be upside down.

    MuJoCo cameras whose world-space 'up' vector (row 1 of cam_mat0) has a
    negative Z component render upside down in rgb_array mode.
    """
    if camera_name is None:
        return False
    model  = env._env.model
    cam_id = model.camera(camera_name).id
    up_z   = model.cam_mat0[cam_id].reshape(3, 3)[1, 2]
    return bool(up_z < 0)


def run_episode(agent, env, deterministic=True, render_mode="human", frame_list=None,
                flip=False, trigger=None, _reset_already_done=None):
    if _reset_already_done is not None:
        obs, info = _reset_already_done
    else:
        obs, info = env.reset()
    if trigger is not None:
        trigger.reset()
    done = False
    ep_reward = 0.0
    ep_success = 0.0
    ep_steps = 0

    while not done:
        if agent is None:
            action = env.action_space.sample()
        elif trigger is not None:
            triggered = trigger.update(obs, info)
            action = agent.select_action(obs, deterministic=deterministic, triggered=triggered)
        else:
            action = agent.select_action(obs, deterministic=deterministic)

        obs, reward, terminated, truncated, info = env.step(action)
        ep_reward  += reward
        ep_success  = max(ep_success, info.get("success", 0.0))
        ep_steps   += 1
        done        = terminated or truncated

        if render_mode == "rgb_array" and frame_list is not None:
            frame = env.render()
            if frame is not None:
                if flip:
                    frame = frame[::-1]
                frame_list.append(frame)
        elif render_mode == "human":
            time.sleep(0.02)  # ~50 fps cap so it's watchable

    return ep_reward, ep_success, ep_steps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name",   type=str, default=None,        help="Experiment run name (looks for final.pt)")
    parser.add_argument("--ckpt",       type=str, default=None,        help="Explicit checkpoint path")
    parser.add_argument("--task",       type=str, default="reach-v3")
    parser.add_argument("--seed",       type=int, default=0)
    parser.add_argument("--n-episodes", type=int, default=50,
                        help="Episodes to run; capped at 50 (number of MT1 task variants)")
    parser.add_argument("--stochastic", action="store_true",           help="Use stochastic policy (default: deterministic)")
    parser.add_argument("--random",     action="store_true",           help="Ignore checkpoint, run random policy")
    parser.add_argument("--save-video", action="store_true",           help="Save video to experiments/<run>/video.mp4")
    parser.add_argument("--camera",     type=str, default=None,
                        help="Camera name: topview, corner, corner2, corner3, behindGripper, gripperPOV")
    args = parser.parse_args()

    render_mode = "rgb_array" if args.save_video else "human"
    env = make_env(args.task, seed=args.seed, render_mode=render_mode)

    # Switch camera if requested.
    # The renderer uses an integer camera_id internally; look it up by name.
    if args.camera is not None:
        cam_id = env._env.model.camera(args.camera).id
        env._env.mujoco_renderer.camera_id = cam_id

    # ---- Load agent ----
    agent   = None
    trigger = None
    if not args.random:
        # Resolve checkpoint path
        if args.ckpt:
            ckpt_path = args.ckpt
        elif args.run_name:
            ckpt_path = os.path.join("experiments", args.run_name, "final.pt")
            # Fall back to latest numbered checkpoint if final.pt doesn't exist yet
            if not os.path.exists(ckpt_path):
                import glob
                ckpts = sorted(glob.glob(os.path.join("experiments", args.run_name, "ckpt_*.pt")))
                if ckpts:
                    ckpt_path = ckpts[-1]
                    print(f"[enjoy] final.pt not found, using latest checkpoint: {ckpt_path}")
                else:
                    print("[enjoy] No checkpoint found — running random policy.")
                    args.random = True
        else:
            print("[enjoy] No --run-name or --ckpt provided — running random policy.")
            args.random = True

        if not args.random:
            # Load config from the run dir to get network shape
            run_dir = os.path.dirname(ckpt_path)
            config_path = os.path.join(run_dir, "config.json")
            if os.path.exists(config_path):
                import json
                with open(config_path) as f:
                    cfg = dict_to_namespace(json.load(f))
            else:
                # Fall back to default config
                with open("configs/sac_baseline.yaml") as f:
                    cfg = dict_to_namespace(yaml.safe_load(f))

            cfg.agent.device = get_device()
            obs_dim    = env.observation_space.shape[0]
            action_dim = env.action_space.shape[0]

            # Use the right agent class depending on config
            if hasattr(cfg, "trigger") and hasattr(cfg, "correction"):
                from src.agents.sac_triggered import SACAgentTriggered
                cfg.agent.correction = cfg.correction
                agent = SACAgentTriggered(obs_dim, action_dim, cfg.agent)
                trigger = make_trigger(cfg.trigger)
            elif hasattr(cfg, "correction"):
                from src.agents.sac_with_correction import SACAgentWithCorrection
                cfg.agent.correction = cfg.correction
                agent = SACAgentWithCorrection(obs_dim, action_dim, cfg.agent)
                trigger = None
            else:
                agent = SACAgent(obs_dim, action_dim, cfg.agent)
                trigger = None

            agent.load(ckpt_path)
            agent.actor.eval()
            print(f"Loaded: {ckpt_path}")

    # ---- Run episodes ----
    all_frames    = [] if args.save_video else None
    deterministic = not args.stochastic
    flip          = needs_vertical_flip(env, args.camera)

    print(f"\nTask: {args.task} | {'random' if args.random else ('stochastic' if args.stochastic else 'deterministic')} policy\n")

    # Default: sweep all 50 MT1 task variants in order (standard eval protocol).
    # Pass --n-episodes N to limit to the first N tasks (useful for quick video recording).
    n_episodes = min(args.n_episodes, env.num_tasks) if not args.random else args.n_episodes

    rewards, successes = [], []
    for ep in range(n_episodes):
        ep_frames = [] if args.save_video else None
        if not args.random:
            # Use reset_to_task so we hit each goal variant exactly once
            obs, info = env.reset_to_task(ep % env.num_tasks)
            r, s, steps = run_episode(agent, env, deterministic, render_mode, ep_frames,
                                      flip=flip, trigger=trigger,
                                      _reset_already_done=(obs, info))
        else:
            r, s, steps = run_episode(agent, env, deterministic, render_mode, ep_frames,
                                      flip=flip, trigger=trigger)
        rewards.append(r)
        successes.append(s)
        if args.save_video and ep_frames:
            all_frames.extend(ep_frames)
        print(f"  Ep {ep+1:3d}  reward={r:8.2f}  success={s:.0f}  steps={steps}")

    print(f"\nMean reward: {np.mean(rewards):.2f}  |  Success rate: {np.mean(successes):.1%}")

    # ---- Save video ----
    if args.save_video and all_frames:
        try:
            import imageio
            out_dir = os.path.join("experiments", args.run_name) if args.run_name else "experiments"
            os.makedirs(out_dir, exist_ok=True)
            cam_suffix = f"_{args.camera}" if args.camera else ""
            video_path = os.path.join(out_dir, f"rollout{cam_suffix}.mp4")
            imageio.mimsave(video_path, all_frames, fps=50)
            print(f"Video saved to {video_path}")
        except ImportError:
            print("imageio not installed — cannot save video. Run: pip install imageio imageio-ffmpeg")

    env.close()


if __name__ == "__main__":
    main()
