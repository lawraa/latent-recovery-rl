"""Evaluate a SACAgentDecoupled checkpoint in two modes:

  1. BASE ONLY   (triggered=False always) — correction module never applied.
     This measures whether gradient isolation kept the base actor independently
     competent.  This is the key number for the co-adaptation paper claim.

  2. ALWAYS CORR (triggered=True  always) — correction applied on every step.
     Sanity-check that the correction module still works after training.

Neither mode uses the heuristic trigger — those eval numbers are already
logged in metrics.jsonl from the training run.

Usage (single run):
    python scripts/eval_base_only.py \\
        --exp-dir modal_experiments/experiments/sac_pp_decoupled_s0

Usage (all seeds, prints a summary table):
    python scripts/eval_base_only.py \\
        modal_experiments/experiments/sac_pp_decoupled_s0 \\
        modal_experiments/experiments/sac_pp_decoupled_s1 \\
        modal_experiments/experiments/sac_pp_decoupled_s2 \\
        modal_experiments/experiments/sac_pp_decoupled_s3
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse, json

import torch

from src.envs.metaworld_wrapper import make_env
from src.agents.sac_decoupled import SACAgentDecoupled
from src.utils.misc import set_seed, get_device, dict_to_namespace


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def _run_eval(agent: SACAgentDecoupled, env, use_correction: bool) -> tuple:
    """Run one full eval sweep (all 50 MT1 variants).

    Args:
        agent:          Loaded SACAgentDecoupled checkpoint.
        env:            MetaWorldWrapper eval environment.
        use_correction: If True,  always pass triggered=True  → correction on.
                        If False, always pass triggered=False → base actor only.

    Returns:
        (mean_reward, success_rate) averaged over all 50 task variants.
    """
    total_reward = total_success = 0.0
    for idx in range(env.num_tasks):
        obs, info = env.reset_to_task(idx)
        done = ep_success = 0.0
        while not done:
            action = agent.select_action(
                obs, deterministic=True, triggered=use_correction
            )
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward  += reward
            ep_success     = max(ep_success, float(info.get("success", 0.0)))
            done           = terminated or truncated
        total_success += ep_success
    n = env.num_tasks
    return total_reward / n, total_success / n


def eval_one(exp_dir: str) -> dict:
    """Load a checkpoint from exp_dir and evaluate in both modes.

    Returns a dict with keys: task, seed, sr_base, sr_corr,
    trained_final_sr (last logged eval/success_rate from metrics.jsonl).
    """
    # ---- Load config -------------------------------------------------------
    config_path = os.path.join(exp_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.json not found in {exp_dir}")
    with open(config_path) as f:
        raw = json.load(f)
    cfg = dict_to_namespace(raw)

    task   = cfg.env.task
    seed   = cfg.training.seed
    device = get_device()

    # Override device to whatever is available on this machine
    cfg.agent.device = device

    # ---- Seed for reproducible eval ----------------------------------------
    set_seed(seed)

    # ---- Build environment -------------------------------------------------
    env = make_env(task, seed=seed)
    obs_dim    = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # ---- Build agent -------------------------------------------------------
    # train_decoupled.py sets cfg.agent.correction before saving config.json,
    # so it is nested. train_with_correction.py saves config.json first and sets
    # cfg.agent.correction after — so always-on checkpoints only have correction
    # at the top level. Handle both.
    if not hasattr(cfg.agent, "correction"):
        cfg.agent.correction = cfg.correction
    agent = SACAgentDecoupled(obs_dim, action_dim, cfg.agent)

    # ---- Load checkpoint ---------------------------------------------------
    # Prefer final.pt; fall back to the latest ckpt_*.pt if the run was
    # stopped early and final.pt was never written.
    ckpt_path = os.path.join(exp_dir, "final.pt")
    if not os.path.exists(ckpt_path):
        candidates = sorted(
            f for f in os.listdir(exp_dir) if f.startswith("ckpt_") and f.endswith(".pt")
        )
        if not candidates:
            raise FileNotFoundError(f"No checkpoint found in {exp_dir}")
        ckpt_path = os.path.join(exp_dir, candidates[-1])
        print(f"  [WARNING] final.pt missing — using latest checkpoint: {candidates[-1]}")
    agent.load(ckpt_path)
    ckpt_name = os.path.basename(ckpt_path)

    print(f"\n[{exp_dir}]  task={task}  seed={seed}  ckpt={ckpt_name}  device={device}")

    # ---- Evaluate: base actor only -----------------------------------------
    print("  → Base actor only (triggered=False) ...")
    _, sr_base = _run_eval(agent, env, use_correction=False)
    print(f"     SR = {sr_base:.2%}")

    # ---- Evaluate: correction always on ------------------------------------
    print("  → Correction always on (triggered=True) ...")
    _, sr_corr = _run_eval(agent, env, use_correction=True)
    print(f"     SR = {sr_corr:.2%}")

    # ---- Read the last logged eval SR from training for reference ----------
    metrics_path = os.path.join(exp_dir, "metrics.jsonl")
    trained_final_sr = None
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            for line in f:
                try:
                    row = json.loads(line)
                    if "eval/success_rate" in row:
                        trained_final_sr = row["eval/success_rate"]
                except json.JSONDecodeError:
                    pass

    env.close()
    return {
        "exp_dir":          exp_dir,
        "task":             task,
        "seed":             seed,
        "ckpt_name":        ckpt_name,
        "sr_base":          sr_base,
        "sr_corr":          sr_corr,
        "trained_final_sr": trained_final_sr,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate decoupled checkpoints: base-only vs always-corrected."
    )
    # Accept either --exp-dir (single) or positional args (multiple)
    parser.add_argument(
        "exp_dirs",
        nargs="*",
        help="One or more experiment folders to evaluate.",
    )
    parser.add_argument(
        "--exp-dir",
        type=str,
        default=None,
        help="Single experiment folder (alternative to positional args).",
    )
    args = parser.parse_args()

    dirs = args.exp_dirs[:]
    if args.exp_dir:
        dirs.insert(0, args.exp_dir)
    if not dirs:
        parser.error("Provide at least one experiment folder.")

    results = []
    for d in dirs:
        try:
            r = eval_one(d)
            results.append(r)
        except Exception as e:
            print(f"\nERROR evaluating {d}: {e}")

    # ---- Summary table -----------------------------------------------------
    if results:
        print("\n" + "=" * 75)
        print("SUMMARY — Co-adaptation ablation: base-only vs always-corrected")
        print("=" * 75)
        print(f"{'Folder':<45} {'Seed':>4}  {'Ckpt':<20}  {'Base only':>9}  {'Always corr':>11}  {'Trained SR':>10}")
        print("-" * 95)
        for r in results:
            folder = os.path.basename(r["exp_dir"])
            trained = f"{r['trained_final_sr']:.2%}" if r["trained_final_sr"] is not None else "  N/A"
            ckpt = r.get("ckpt_name", "?")
            print(
                f"{folder:<45} {r['seed']:>4}  {ckpt:<20}  "
                f"{r['sr_base']:>8.2%}  {r['sr_corr']:>10.2%}  {trained:>10}"
            )
        print("-" * 95)
        mean_base = sum(r["sr_base"] for r in results) / len(results)
        mean_corr = sum(r["sr_corr"] for r in results) / len(results)
        print(f"{'MEAN':<45} {'':>4}  {'':20}  {mean_base:>8.2%}  {mean_corr:>10.2%}")
        print("=" * 95)
        print()
        print("Key: 'Base only' = correction DISABLED (triggered=False always)")
        print("     'Always corr' = correction ENABLED  (triggered=True  always)")
        print("     'Trained SR'  = last logged eval/success_rate from training")
        print()
        if mean_base > 0.5:
            print("✓ Base actor is independently competent (SR > 50% without correction).")
            print("  Gradient isolation preserved base actor quality as expected.")
        else:
            print("✗ Base actor degraded without correction (SR < 50%).")
            print("  This may indicate co-adaptation — check always-on runs for comparison.")


if __name__ == "__main__":
    main()
