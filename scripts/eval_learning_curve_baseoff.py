"""Multi-checkpoint base-off analysis for the decoupled model.

For each checkpoint asks: does triggered correction outperform the base actor
alone *during* learning, before the base reaches ceiling?

Evaluates two modes at every requested checkpoint:
  base_only  — base actor forward, no correction
  triggered  — correction applied only when NearFailureTrigger fires

Outputs a table of:  step | base | triggered | gap | trigger_rate

Usage:
    # Default: a curated set spanning the learning curve
    python scripts/eval_learning_curve_baseoff.py

    # Explicit steps
    python scripts/eval_learning_curve_baseoff.py --steps 600000 900000 1350000 1500000 2500000

    # Every saved checkpoint
    python scripts/eval_learning_curve_baseoff.py --all

    # Different run directory
    python scripts/eval_learning_curve_baseoff.py --run-dir experiments/sac_pp_decoupled_s0
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import glob
import json
import yaml
import numpy as np
import torch

from src.envs.metaworld_wrapper import make_env
from src.agents.sac_decoupled import SACAgentDecoupled
from src.agents.sac_with_correction import SACAgentWithCorrection
from src.triggers.near_failure import make_trigger
from src.utils.misc import set_seed, get_device, dict_to_namespace


# Key checkpoints that span the three learning phases:
#   pre-success   (trigger active, base near-zero)
#   learning phase (trigger active, base climbing — the critical window)
#   convergence   (base saturated, correction marginal)
DEFAULT_STEPS = [
    500_000,
    600_000,
    700_000,
    900_000,
    1_000_000,
    1_150_000,
    1_200_000,
    1_350_000,
    1_500_000,
    1_700_000,
    2_000_000,
    2_500_000,
]


# ---------------------------------------------------------------------------
# Single-mode eval loop
# ---------------------------------------------------------------------------

def evaluate_mode(agent, env, mode: str, trigger_cfg=None):
    """Run all 50 MT1 variants in a fixed mode.

    mode:
      "base_only"    — base actor forward, no correction (works for any agent type)
      "triggered"    — correction gated by NearFailureTrigger (SACAgentDecoupled)
      "corrected_on" — correction always active (SACAgentWithCorrection normal mode)

    trigger_cfg is only required for mode="triggered".
    Returns (success_rate, mean_reward, trigger_rate)
    """
    assert mode in ("base_only", "triggered", "corrected_on")
    if mode == "triggered":
        assert trigger_cfg is not None, "trigger_cfg required for mode='triggered'"
    n             = env.num_tasks
    total_success = 0.0
    total_reward  = 0.0
    total_trig    = 0.0
    device        = agent.device

    for idx in range(n):
        obs, info  = env.reset_to_task(idx)
        trigger    = make_trigger(trigger_cfg) if trigger_cfg else None
        done       = False
        ep_success = 0.0

        while not done:
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)

            if mode == "base_only":
                with torch.no_grad():
                    action = agent.actor.get_action(obs_t, deterministic=True).squeeze(0)
            elif mode == "triggered":
                fired  = trigger.update(obs, info)
                action = agent.select_action(obs, deterministic=True, triggered=fired)
            else:  # corrected_on — always-on agent's normal select_action
                action = agent.select_action(obs, deterministic=True)

            obs, reward, terminated, truncated, info = env.step(action)
            ep_success  = max(ep_success, info.get("success", 0.0))
            done        = terminated or truncated
            total_reward += reward

        total_success += ep_success
        if trigger is not None:
            total_trig += trigger.fire_rate

    return total_success / n, total_reward / n, total_trig / n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        default="experiments/sac_pp_decoupled_v1",
        help="Experiment directory containing ckpt_*.pt files",
    )
    parser.add_argument(
        "--config",
        default="configs/pick_place_decoupled.yaml",
    )
    parser.add_argument(
        "--agent-type",
        default="decoupled",
        choices=["decoupled", "always_on"],
        help="'decoupled' (default) compares base vs triggered. "
             "'always_on' compares base vs corrected_on.",
    )
    parser.add_argument(
        "--also-corrected-on",
        action="store_true",
        help="For decoupled agent: also evaluate correction always-ON as a third column.",
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        type=int,
        default=None,
        help="Explicit list of steps to evaluate. Overrides --all and default set.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Evaluate every ckpt_*.pt found in --run-dir",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--save-json",
        type=str,
        default=None,
        help="Write results to this JSON path",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()

    with open(args.config) as f:
        cfg = dict_to_namespace(yaml.safe_load(f))
    cfg.agent.device     = device
    cfg.agent.correction = cfg.correction

    eval_env   = make_env(cfg.env.task, seed=args.seed)
    obs_dim    = eval_env.observation_space.shape[0]
    action_dim = eval_env.action_space.shape[0]
    print(f"Task: {cfg.env.task}  obs_dim={obs_dim}  action_dim={action_dim}  "
          f"num_tasks={eval_env.num_tasks}  device={device}\n")

    # Determine which steps to run
    if args.steps:
        steps_to_run = sorted(args.steps)
    elif args.all:
        pattern = os.path.join(args.run_dir, "ckpt_*.pt")
        found = sorted(glob.glob(pattern))
        steps_to_run = []
        for p in found:
            name = os.path.basename(p)          # ckpt_0600000.pt
            try:
                steps_to_run.append(int(name.replace("ckpt_", "").replace(".pt", "")))
            except ValueError:
                pass
    else:
        steps_to_run = DEFAULT_STEPS

    # Choose agent class and modes to evaluate
    if args.agent_type == "always_on":
        AgentClass   = SACAgentWithCorrection
        second_mode  = "corrected_on"
        col2_label   = "corrected"
        run_third    = False
    else:
        AgentClass   = SACAgentDecoupled
        second_mode  = "triggered"
        col2_label   = "triggered"
        run_third    = args.also_corrected_on

    n_modes = 3 if run_third else 2
    print(f"Will evaluate {len(steps_to_run)} checkpoints × {n_modes} modes × "
          f"{eval_env.num_tasks} tasks = "
          f"{len(steps_to_run) * n_modes * eval_env.num_tasks} episodes total\n")

    if run_third:
        header = (
            f"{'step':>10}  {'base':>7}  {col2_label:>9}  {'always_on':>9}  "
            f"{'gap_trig':>9}  {'gap_corr':>9}  {'trig_rate':>10}"
        )
    else:
        header = (
            f"{'step':>10}  {'base':>7}  {col2_label:>9}  {'gap':>7}  "
            f"{'trig_rate':>10}  {'reward_base':>12}  {'reward_2nd':>12}"
        )
    divider = "-" * len(header)
    print(header)
    print(divider)

    all_results = []

    for step in steps_to_run:
        ckpt_path = os.path.join(args.run_dir, f"ckpt_{step:07d}.pt")
        if not os.path.exists(ckpt_path):
            ckpt_path = os.path.join(args.run_dir, f"ckpt_{step}.pt")
        if not os.path.exists(ckpt_path):
            print(f"  [SKIP] No checkpoint found for step {step}")
            continue

        agent = AgentClass(obs_dim, action_dim, cfg.agent)
        agent.load(ckpt_path)

        trigger_cfg = getattr(cfg, "trigger", None)
        base_succ, base_rew, _         = evaluate_mode(agent, eval_env, "base_only", trigger_cfg)
        trig_succ, trig_rew, trig_rate = evaluate_mode(agent, eval_env, second_mode, trigger_cfg)

        row = {
            "step":         step,
            "base":         base_succ,
            "triggered":    trig_succ,
            "gap":          trig_succ - base_succ,
            "trigger_rate": trig_rate,
            "reward_base":  base_rew,
            "reward_trig":  trig_rew,
        }

        if run_third:
            corr_succ, corr_rew, _ = evaluate_mode(agent, eval_env, "corrected_on", trigger_cfg)
            row["corrected_on"] = corr_succ
            row["gap_corr"]     = corr_succ - base_succ
            gap_t = f"{trig_succ - base_succ:+.1%}" if trig_succ != base_succ else "   0.0%"
            gap_c = f"{corr_succ - base_succ:+.1%}" if corr_succ != base_succ else "   0.0%"
            print(
                f"{step:>10,}  "
                f"{base_succ:>7.1%}  "
                f"{trig_succ:>9.1%}  "
                f"{corr_succ:>9.1%}  "
                f"{gap_t:>9}  "
                f"{gap_c:>9}  "
                f"{trig_rate:>10.2%}"
            )
        else:
            gap_str = f"{trig_succ - base_succ:+.1%}" if trig_succ != base_succ else "   0.0%"
            print(
                f"{step:>10,}  "
                f"{base_succ:>7.1%}  "
                f"{trig_succ:>9.1%}  "
                f"{gap_str:>7}  "
                f"{trig_rate:>10.2%}  "
                f"{base_rew:>12.1f}  "
                f"{trig_rew:>12.1f}"
            )

        all_results.append(row)

    print(divider)

    # Summary: learning-phase rows where gap > 0
    learning_rows = [r for r in all_results if r["gap"] > 0]
    if learning_rows:
        max_gap_row = max(learning_rows, key=lambda r: r["gap"])
        print(f"\n  Peak gap: step {max_gap_row['step']:,}  "
              f"base={max_gap_row['base']:.1%}  "
              f"triggered={max_gap_row['triggered']:.1%}  "
              f"gap={max_gap_row['gap']:+.1%}")
        print(f"  Steps where triggered > base: "
              f"{[r['step'] for r in learning_rows]}")
    else:
        print("\n  No steps found where triggered > base.")

    if args.save_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_json)), exist_ok=True)
        with open(args.save_json, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {args.save_json}")

    eval_env.close()


if __name__ == "__main__":
    main()
