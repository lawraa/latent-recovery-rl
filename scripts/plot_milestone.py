"""Generate training curve figure for milestone report.

Plots eval success rate vs training steps for SAC baseline and
SAC + Decoupled Latent Correction, 4 seeds each on pick-place-v3.

Output: reports/figures/training_curves.png
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Data paths ────────────────────────────────────────────────────────────────
RUNS = {
    "SAC Baseline": [
        "experiments/sac_pp_baseline_s0_pilot",
        "experiments/sac_pp_baseline_s1_pilot",
        "modal_experiments/experiments/sac_pp_baseline_s2",
        "modal_experiments/experiments/sac_pp_baseline_s3",
    ],
    "SAC + Decoupled\nLatent Correction": [
        "experiments/sac_pp_decoupled_s0",
        "experiments/sac_pp_decoupled_s1",
        "modal_experiments/experiments/sac_pp_decoupled_s2",
        "modal_experiments/experiments/sac_pp_decoupled_s3",
    ],
}

COLORS = {
    "SAC Baseline":                    "#E07B54",
    "SAC + Decoupled\nLatent Correction": "#5B8DB8",
}

EVAL_STEPS = np.arange(50_000, 2_500_001, 50_000)   # 50 evals


def load_eval_curve(path):
    """Return (steps, success_rates) arrays from metrics.jsonl."""
    steps, srs = [], []
    with open(f"{path}/metrics.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if "eval/success_rate" in r:
                steps.append(r["step"])
                srs.append(r["eval/success_rate"] * 100.0)
    return np.array(steps), np.array(srs)


def interpolate_to_grid(steps, values, grid):
    """Linearly interpolate a curve onto a common step grid."""
    return np.interp(grid, steps, values)


# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))

for label, paths in RUNS.items():
    color = COLORS[label]
    curves = []
    for path in paths:
        steps, srs = load_eval_curve(path)
        curve = interpolate_to_grid(steps, srs, EVAL_STEPS)
        curves.append(curve)
        ax.plot(EVAL_STEPS / 1e6, curve,
                color=color, alpha=0.25, linewidth=1.0)

    arr  = np.array(curves)
    mean = arr.mean(axis=0)
    ax.plot(EVAL_STEPS / 1e6, mean,
            color=color, linewidth=2.5, label=label)
    ax.fill_between(EVAL_STEPS / 1e6,
                    arr.min(axis=0), arr.max(axis=0),
                    color=color, alpha=0.12)

# ── Formatting ────────────────────────────────────────────────────────────────
ax.set_xlabel("Training Steps (millions)", fontsize=12)
ax.set_ylabel("Eval Success Rate (%)", fontsize=12)
ax.set_title("pick-place-v3 · 4 seeds each", fontsize=12)
ax.set_xlim(0, 2.5)
ax.set_ylim(-2, 105)
ax.set_xticks([0, 0.5, 1.0, 1.5, 2.0, 2.5])
ax.set_yticks([0, 20, 40, 60, 80, 100])
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x)}%"))
ax.grid(True, alpha=0.3, linestyle="--")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Custom legend: thick line + shaded band
legend_handles = []
for label, color in COLORS.items():
    patch = mpatches.Patch(facecolor=color, alpha=0.3, edgecolor=color)
    line  = plt.Line2D([0], [0], color=color, linewidth=2.5)
    legend_handles.append((line, patch, label))

ax.legend(
    handles=[plt.Line2D([0], [0], color=c, linewidth=2.5,
                        label=l.replace("\n", " "))
             for l, c in COLORS.items()],
    loc="upper left", fontsize=10, framealpha=0.9,
)

# Annotate final mean values
for label, paths in RUNS.items():
    color = COLORS[label]
    curves = []
    for path in paths:
        steps, srs = load_eval_curve(path)
        curves.append(interpolate_to_grid(steps, srs, EVAL_STEPS))
    mean_final = np.array(curves).mean(axis=0)[-1]
    ax.annotate(f"{mean_final:.0f}%",
                xy=(2.5, mean_final),
                xytext=(2.52, mean_final),
                fontsize=9, color=color, va="center",
                annotation_clip=False)

plt.tight_layout()
out = "reports/figures/training_curves.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
