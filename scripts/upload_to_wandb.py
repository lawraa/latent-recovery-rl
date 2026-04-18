"""Replay a completed run's metrics.jsonl into wandb.

Reads config.json and metrics.jsonl from an experiment directory and logs
every record to wandb exactly as the training script would have, preserving
step numbers and all metric keys.

Usage:
    # Upload specific runs
    python scripts/upload_to_wandb.py --runs sac_push_baseline_s0 sac_push_decoupled_s0

    # Upload all four runs with no args (default list)
    python scripts/upload_to_wandb.py

Searches experiments/ first, then modal_experiments/experiments/.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import json

import wandb

SEARCH_ROOTS = [
    "experiments",
    "modal_experiments/experiments",
]

DEFAULT_RUNS = [
    "sac_push_baseline_s0",
    "sac_push_decoupled_s0",
    "sac_pp_action_residual_s0",
    "sac_pp_action_residual_s1",
]


def find_run_dir(run_name: str) -> str:
    for root in SEARCH_ROOTS:
        candidate = os.path.join(root, run_name)
        if os.path.isdir(candidate):
            return candidate
    raise FileNotFoundError(
        f"Run '{run_name}' not found in any of: {SEARCH_ROOTS}"
    )


def upload_run(run_name: str):
    run_dir = find_run_dir(run_name)

    config_path  = os.path.join(run_dir, "config.json")
    metrics_path = os.path.join(run_dir, "metrics.jsonl")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Missing config.json in {run_dir}")
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"Missing metrics.jsonl in {run_dir}")

    with open(config_path) as f:
        config = json.load(f)

    project  = config.get("logging", {}).get("project", "latent-recovery-rl")

    print(f"\n{'='*60}")
    print(f"  Uploading: {run_name}")
    print(f"  Source:    {run_dir}")
    print(f"  Project:   {project}")
    print(f"{'='*60}")

    wandb.init(
        project  = project,
        name     = run_name,
        config   = config,
        dir      = run_dir,
        reinit   = True,
    )

    with open(metrics_path) as f:
        lines = f.readlines()

    total = len(lines)
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        step   = record.pop("step")
        wandb.log(record, step=step)
        if (i + 1) % 500 == 0 or (i + 1) == total:
            print(f"  {i+1:>6}/{total} records logged (step={step})")

    wandb.finish()
    print(f"  Done: {run_name}\n")


def main():
    parser = argparse.ArgumentParser(description="Replay metrics.jsonl into wandb")
    parser.add_argument(
        "--runs",
        nargs="*",
        default=DEFAULT_RUNS,
        help="Run names to upload (default: the 4 runs missing wandb data)",
    )
    args = parser.parse_args()

    failed = []
    for run_name in args.runs:
        try:
            upload_run(run_name)
        except Exception as e:
            print(f"  ERROR uploading {run_name}: {e}")
            failed.append(run_name)

    if failed:
        print(f"\nFailed runs: {failed}")
        sys.exit(1)
    else:
        print("All runs uploaded successfully.")


if __name__ == "__main__":
    main()
