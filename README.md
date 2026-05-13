<div align="center">

# Near-Failure Latent Corrections as a Training Curriculum for Deep RL

**Shou-Jen Chen** · UC Berkeley EECS · CS285 Deep Reinforcement Learning

[![Paper](https://img.shields.io/badge/Paper-CS285_Final_Project-red?style=for-the-badge&logo=arxiv)](https://github.com/lawraa/latent-recovery-rl)
[![Meta-World](https://img.shields.io/badge/Benchmark-Meta--World_3.0-green?style=for-the-badge)](https://github.com/Farama-Foundation/Metaworld)

</div>

---

## Overview

We propose training a small latent correction module concurrently with an SAC base actor on robot manipulation tasks. A lightweight near-failure heuristic detects when the policy stalls at bottleneck states (e.g., pre-grasp approach, puck transport), and substitutes a corrected action during rollout. The resulting transitions enter the replay buffer normally, producing stronger Bellman targets at stall states and improving the base actor's critic signal — without the base actor receiving any direct gradient from the correction module.

The key challenge is **co-adaptation**: naive concurrent training causes the actor encoder to learn correction-compatible representations rather than independently good actions, collapsing from 96% to 36% success when the correction is removed. A **stop-gradient on the encoder** during the correction update severs this gradient path, enabling reliable multi-seed convergence while keeping the base actor independently deployable.

<div align="center">

| Task | Baseline SAC | Ours (Decoupled) |
|------|-------------|-----------------|
| pick-place-v3 | 66.0% mean final SR | **93.5%** mean final SR |
| push-v3 | 32.5% mean final SR | **89.5%** mean final SR |

*4 seeds each, 2.5M steps, Meta-World 3.0 MT1 protocol*

</div>

---

## Method

The base actor and correction module are trained **concurrently from scratch** using three separate update steps per iteration:

1. **Critic** — standard SAC Bellman target using base policy actions
2. **Base actor** — standard SAC actor loss, no correction in the computation graph
3. **Correction** — SAC-style loss on the corrected policy, with `torch.no_grad()` on the encoder so no gradients reach actor parameters

The near-failure trigger uses geometric signals (hand-to-object distance, object-to-goal distance) with a sliding-window stall detector. It fires heavily early in training when the policy stalls frequently, and self-regulates to below 15% of steps at convergence — with no manual annealing.

<div align="center">
<img src="docs/main_paper/CS285_Final_Project/figures/fig6_trigger_rate.pdf" width="70%" alt="Trigger rate self-regulates over training">
</div>

---

## Installation

```bash
# Create and activate environment
conda create -n latent-recovery python=3.10
conda activate latent-recovery

# Install Meta-World 3.0
pip install git+https://github.com/Farama-Foundation/Metaworld.git@main

# Install remaining dependencies
pip install -r requirements.txt

# Install package in editable mode
pip install -e .
```

> **Note:** Tested on Apple MPS (M-series) and NVIDIA A100. MuJoCo 3.x required.

---

## Reproducing Results

All training commands require `--wandb` for logging.

### Baseline SAC

```bash
# Pick-place-v3 (run for seeds 0–3)
python scripts/train_sac.py \
    --config configs/pick_place_baseline.yaml \
    --seed 0 --run-name sac_pp_baseline_s0 --wandb

# Push-v3
python scripts/train_sac.py \
    --config configs/push_baseline.yaml \
    --seed 0 --run-name sac_push_baseline_s0 --wandb
```

### Decoupled Correction (Ours)

```bash
# Pick-place-v3
python scripts/train_decoupled.py \
    --config configs/pick_place_decoupled.yaml \
    --seed 0 --run-name sac_pp_decoupled_s0 --wandb

# Push-v3
python scripts/train_decoupled.py \
    --config configs/push_decoupled.yaml \
    --seed 0 --run-name sac_push_decoupled_s0 --wandb
```

### Ablations

```bash
# HER comparison (push-v3)
python scripts/train_her.py \
    --config configs/push_her.yaml \
    --seed 0 --run-name sac_push_her_s0 --wandb

# Action-space residual (push-v3)
python scripts/train_action_residual.py \
    --config configs/push_action_residual.yaml \
    --seed 0 --run-name sac_push_ar_s0 --wandb

# Sequential vs concurrent (pick-place-v3)
# Requires a converged baseline checkpoint at experiments/sac_pp_baseline_s0/final.pt
python scripts/train_sequential_correction.py \
    --config configs/pick_place_sequential_correction.yaml \
    --seed 0 --run-name sac_pp_seq_s0 --wandb
```

### Base-off Evaluation (co-adaptation check)

```bash
python scripts/eval_base_only.py \
    --checkpoint experiments/sac_pp_decoupled_s0/final.pt \
    --config configs/pick_place_decoupled.yaml
```

### Remote Training (Modal A100)

```bash
# Baseline
uv run modal run --detach scripts/modal_train.py::train_sac -- \
    --config configs/pick_place_baseline.yaml --seed 0 \
    --run-name sac_pp_baseline_s0 --wandb

# Decoupled
uv run modal run --detach scripts/modal_train.py::train_decoupled -- \
    --config configs/pick_place_decoupled.yaml --seed 0 \
    --run-name sac_pp_decoupled_s0 --wandb

# Download results
uv run modal volume get latent-recovery-rl-volume \
    /experiments/<run_name> modal_experiments/experiments/
```

---

## Repository Structure

```
├── configs/                  # YAML configs for all experiments
├── scripts/
│   ├── train_sac.py          # Baseline SAC
│   ├── train_decoupled.py    # Decoupled correction (main method)
│   ├── train_with_correction.py  # Always-on joint training (co-adaptation diagnostic)
│   ├── train_action_residual.py  # Action-space residual ablation
│   ├── train_her.py          # HER comparison
│   ├── train_sequential_correction.py  # Sequential vs concurrent ablation
│   ├── modal_train.py        # Modal remote training
│   ├── eval_base_only.py     # Base-off evaluation
│   └── eval_base_off.py      # Always-on vs decoupled comparison
├── src/
│   ├── agents/
│   │   ├── sac.py                       # SAC baseline
│   │   ├── sac_decoupled.py             # Decoupled correction agent
│   │   ├── sac_with_correction.py       # Always-on joint training
│   │   ├── sac_action_residual.py       # Action-space residual agent
│   │   └── sac_sequential_correction.py # Sequential correction agent
│   ├── modules/
│   │   ├── latent_correction.py  # f(s, z) → Δz correction MLP
│   │   └── action_residual.py    # Action-space residual MLP
│   ├── triggers/
│   │   └── near_failure.py       # Heuristic stall detectors (per task)
│   ├── envs/
│   │   └── metaworld_wrapper.py  # Meta-World MT1 environment wrapper
│   └── utils/                    # Replay buffer, logging utilities
└── tests/
    └── test_triggered_pipeline.py
```

---

## Key Results

**Gradient isolation prevents co-adaptation.** Without the stop-gradient, concurrent training causes encoder co-adaptation: the base actor drops from 96% to 36% when the correction is removed. With gradient isolation, base-off SR equals triggered SR across all seeds (pick-place mean 95% base-only vs 93.5% triggered).

**Trigger specificity matters.** A random trigger at 5% fire rate performs no better than the baseline (36% vs 32.5% mean final SR on push-v3), while the heuristic trigger reaches 89.5%.

**Concurrent training is necessary.** Sequential correction (load converged baseline, freeze actor, train correction) achieves 65.0% mean final SR — below the frozen baseline at 66.0% — and fails to rescue any seed the baseline could not solve, while using 60% more compute.

---

## Citation

```bibtex
@misc{chen2025latentcorrection,
  title   = {Near-Failure Latent Corrections as a Training Curriculum for Deep RL},
  author  = {Chen, Shou-Jen},
  year    = {2026},
  note    = {CS285 Deep Reinforcement Learning Final Project, UC Berkeley}
}
```
