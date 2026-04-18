"""Print a live tail of eval metrics from a running or finished experiment.

Usage:
    python scripts/monitor.py --run-name sac_reach_s0
    python scripts/monitor.py --run-name sac_reach_s0 --plot   # also show ASCII chart
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import json
import time


def load_records(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def ascii_chart(values, width=60, height=10, label=""):
    if not values:
        return
    lo, hi = min(values), max(values)
    if hi == lo:
        hi = lo + 1
    buckets = [[] for _ in range(width)]
    for i, v in enumerate(values):
        idx = min(int((i / len(values)) * width), width - 1)
        buckets[idx].append(v)
    cols = [(sum(b) / len(b) if b else None) for b in buckets]
    print(f"\n  {label}  [min={lo:.3f}  max={hi:.3f}]")
    for row in range(height, 0, -1):
        threshold = lo + (row / height) * (hi - lo)
        line = "".join("█" if (c is not None and c >= threshold) else " " for c in cols)
        print(f"  |{line}|")
    print(f"  +{'-'*width}+")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--plot",     action="store_true", help="Print ASCII charts")
    parser.add_argument("--follow",   action="store_true", help="Keep watching (like tail -f)")
    args = parser.parse_args()

    metrics_path = os.path.join("experiments", args.run_name, "metrics.jsonl")
    if not os.path.exists(metrics_path):
        print(f"Not found: {metrics_path}")
        sys.exit(1)

    while True:
        records = load_records(metrics_path)
        eval_records = [r for r in records if "eval/success_rate" in r]
        train_records = [r for r in records if "train/episode_reward" in r]

        os.system("clear")
        print(f"=== {args.run_name} ===")
        print(f"  Total log entries: {len(records)}")
        if train_records:
            last = train_records[-1]
            print(f"  Last train step:   {last.get('step', '?'):,}")
            print(f"  Last ep reward:    {last.get('train/episode_reward', '?'):.2f}")
        if eval_records:
            last_eval = eval_records[-1]
            print(f"\n  Eval results ({len(eval_records)} evals so far):")
            print(f"    step={last_eval.get('step', '?'):,}  "
                  f"reward={last_eval.get('eval/mean_reward', '?'):.2f}  "
                  f"success={last_eval.get('eval/success_rate', '?'):.1%}")
            print()
            print("  All eval success rates:")
            for r in eval_records[-10:]:  # last 10
                bar = "█" * int(r.get("eval/success_rate", 0) * 40)
                print(f"    step {r.get('step', '?'):>7,}  [{bar:<40}]  {r.get('eval/success_rate', 0):.1%}")
        else:
            print("\n  No eval results yet (eval runs every 10k steps by default).")

        if args.plot and eval_records:
            ascii_chart(
                [r.get("eval/success_rate", 0) for r in eval_records],
                label="eval/success_rate"
            )

        if not args.follow:
            break
        time.sleep(10)


if __name__ == "__main__":
    main()
