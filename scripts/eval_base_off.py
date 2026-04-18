"""Base-off ablation evaluation.

Tests whether the always-on model's base actor has co-adapted (and collapses
without the correction) versus the decoupled model's base actor (which should
remain strong independently).

Evaluates six conditions over all 50 MT1 task variants each:

  baseline_base       — Baseline SAC, no correction (reference)
  always_on_corrected — Always-on model, correction ON  (its normal mode)
  always_on_base      — Always-on model, correction OFF (the co-adaptation test)
  decoupled_base      — Decoupled model, correction OFF (base actor independently)
  decoupled_triggered — Decoupled model, trigger-gated  (its normal mode)
  decoupled_always_on — Decoupled model, correction always ON (upper bound)

Usage:
    python scripts/eval_base_off.py
    python scripts/eval_base_off.py --always-on-ckpt experiments/sac_pp_latent_always_v2/final.pt
    python scripts/eval_base_off.py --decoupled-ckpt experiments/sac_pp_decoupled_v1/ckpt_2350000.pt
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import json
import yaml
import numpy as np
import torch

from src.envs.metaworld_wrapper import make_env
from src.agents.sac import SACAgent
from src.agents.sac_with_correction import SACAgentWithCorrection
from src.agents.sac_decoupled import SACAgentDecoupled
from src.triggers.near_failure import make_trigger
from src.utils.misc import set_seed, get_device, dict_to_namespace


# ---------------------------------------------------------------------------
# Action-selection helpers
# ---------------------------------------------------------------------------

def _base_action(agent, obs_t: torch.Tensor) -> np.ndarray:
    """Run the base actor only, bypassing any correction module."""
    with torch.no_grad():
        return agent.actor.get_action(obs_t, deterministic=True).squeeze(0)


def _corrected_action(agent, obs_t: torch.Tensor) -> np.ndarray:
    """Run the corrected path unconditionally (always-on mode)."""
    with torch.no_grad():
        z       = agent.actor.trunk(obs_t)
        delta_z = agent.correction(obs_t, z)
        z_corr  = z + delta_z
        action  = torch.tanh(agent.actor.mean_head(z_corr))
    return action.cpu().numpy().squeeze(0)


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def evaluate(agent, env, mode: str, trigger_cfg=None):
    """Evaluate agent over all 50 MT1 task variants.

    mode options
    ------------
    "base_only"     — base actor forward, no correction ever
    "correction_on" — corrected path always active
    "triggered"     — corrected path only when trigger fires (requires trigger_cfg)

    Returns dict: success_rate, mean_reward, mean_trigger_rate, delta_z_mean
    """
    assert mode in ("base_only", "correction_on", "triggered"), f"Unknown mode: {mode}"
    if mode == "triggered":
        assert trigger_cfg is not None, "trigger_cfg required for mode='triggered'"

    n              = env.num_tasks
    total_reward   = 0.0
    total_success  = 0.0
    total_trig     = 0.0
    total_delta_z  = 0.0
    delta_z_steps  = 0

    device = agent.device

    for idx in range(n):
        obs, info = env.reset_to_task(idx)
        trigger   = make_trigger(trigger_cfg) if trigger_cfg else None
        done      = False
        ep_success = 0.0

        while not done:
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)

            if mode == "base_only":
                action = _base_action(agent, obs_t)

            elif mode == "correction_on":
                action = _corrected_action(agent, obs_t)

            elif mode == "triggered":
                fired = trigger.update(obs, info)
                # SACAgentDecoupled.select_action already handles triggered flag
                action = agent.select_action(obs, deterministic=True, triggered=fired)

            obs, reward, terminated, truncated, info = env.step(action)
            ep_success  = max(ep_success, info.get("success", 0.0))
            done        = terminated or truncated
            total_reward += reward

            # Track Δz magnitude when correction is active
            if mode in ("correction_on", "triggered") and hasattr(agent, "correction"):
                with torch.no_grad():
                    obs_t2 = torch.FloatTensor(obs).unsqueeze(0).to(device)
                    z      = agent.actor.trunk(obs_t2)
                    dz     = agent.correction(obs_t2, z)
                    total_delta_z += dz.norm(dim=-1).mean().item()
                    delta_z_steps += 1

        total_success += ep_success
        if trigger is not None:
            total_trig += trigger.fire_rate

    return {
        "success_rate":    total_success / n,
        "mean_reward":     total_reward  / n,
        "trigger_rate":    total_trig    / n if trigger_cfg else 0.0,
        "delta_z_mean":    total_delta_z / delta_z_steps if delta_z_steps > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# Pretty print
# ---------------------------------------------------------------------------

def _row(label, res):
    return (
        f"  {label:<28s}  "
        f"success={res['success_rate']:.1%}  "
        f"reward={res['mean_reward']:8.1f}  "
        f"trig_rate={res['trigger_rate']:.2%}  "
        f"delta_z={res['delta_z_mean']:.2f}"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline-ckpt",
        default="experiments/sac_pp_baseline_v2/final.pt",
        help="Path to baseline SAC final checkpoint",
    )
    parser.add_argument(
        "--always-on-ckpt",
        default="experiments/sac_pp_latent_always_v2/final.pt",
        help="Path to always-on correction final checkpoint",
    )
    parser.add_argument(
        "--decoupled-ckpt",
        default="experiments/sac_pp_decoupled_v1/final.pt",
        help="Path to decoupled model final checkpoint",
    )
    parser.add_argument(
        "--always-on-config",
        default="configs/pick_place_latent_always.yaml",
    )
    parser.add_argument(
        "--decoupled-config",
        default="configs/pick_place_decoupled.yaml",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-json", type=str, default=None,
                        help="If set, write results to this JSON file")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device}")

    # --- Build eval environment ---
    eval_env = make_env("pick-place-v3", seed=args.seed)
    obs_dim    = eval_env.observation_space.shape[0]
    action_dim = eval_env.action_space.shape[0]
    print(f"obs_dim={obs_dim}  action_dim={action_dim}  num_tasks={eval_env.num_tasks}\n")

    results = {}

    # ------------------------------------------------------------------
    # 1. Baseline SAC (reference — no correction, base actor only)
    # ------------------------------------------------------------------
    print("Loading baseline SAC...")
    with open("configs/sac_baseline.yaml") as f:
        base_cfg = dict_to_namespace(yaml.safe_load(f))
    base_cfg.agent.device = device
    baseline_agent = SACAgent(obs_dim, action_dim, base_cfg.agent)
    baseline_agent.load(args.baseline_ckpt)

    print("  Evaluating baseline_base ... ", end="", flush=True)
    res = evaluate(baseline_agent, eval_env, mode="base_only")
    results["baseline_base"] = res
    print(f"success={res['success_rate']:.1%}")

    # ------------------------------------------------------------------
    # 2. Always-on correction
    # ------------------------------------------------------------------
    print("\nLoading always-on correction model...")
    with open(args.always_on_config) as f:
        ao_cfg = dict_to_namespace(yaml.safe_load(f))
    ao_cfg.agent.device     = device
    ao_cfg.agent.correction = ao_cfg.correction
    ao_agent = SACAgentWithCorrection(obs_dim, action_dim, ao_cfg.agent)
    ao_agent.load(args.always_on_ckpt)

    print("  Evaluating always_on_corrected ... ", end="", flush=True)
    res = evaluate(ao_agent, eval_env, mode="correction_on")
    results["always_on_corrected"] = res
    print(f"success={res['success_rate']:.1%}")

    print("  Evaluating always_on_base (correction OFF) ... ", end="", flush=True)
    res = evaluate(ao_agent, eval_env, mode="base_only")
    results["always_on_base"] = res
    print(f"success={res['success_rate']:.1%}")

    # ------------------------------------------------------------------
    # 3. Decoupled model
    # ------------------------------------------------------------------
    print("\nLoading decoupled model...")
    with open(args.decoupled_config) as f:
        dc_cfg = dict_to_namespace(yaml.safe_load(f))
    dc_cfg.agent.device     = device
    dc_cfg.agent.correction = dc_cfg.correction
    dc_agent = SACAgentDecoupled(obs_dim, action_dim, dc_cfg.agent)
    dc_agent.load(args.decoupled_ckpt)

    print("  Evaluating decoupled_base (correction OFF) ... ", end="", flush=True)
    res = evaluate(dc_agent, eval_env, mode="base_only")
    results["decoupled_base"] = res
    print(f"success={res['success_rate']:.1%}")

    print("  Evaluating decoupled_triggered (normal mode) ... ", end="", flush=True)
    res = evaluate(dc_agent, eval_env, mode="triggered", trigger_cfg=dc_cfg.trigger)
    results["decoupled_triggered"] = res
    print(f"success={res['success_rate']:.1%}")

    print("  Evaluating decoupled_always_on (correction always ON) ... ", end="", flush=True)
    res = evaluate(dc_agent, eval_env, mode="correction_on")
    results["decoupled_always_on"] = res
    print(f"success={res['success_rate']:.1%}")

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    print("\n" + "=" * 75)
    print("  BASE-OFF ABLATION — pick-place-v3 — 50 task variants each")
    print("=" * 75)
    order = [
        ("baseline_base",        "baseline_base       "),
        ("always_on_corrected",  "always_on_corrected "),
        ("always_on_base",       "always_on_base [!]  "),
        ("decoupled_base",       "decoupled_base      "),
        ("decoupled_triggered",  "decoupled_triggered "),
        ("decoupled_always_on",  "decoupled_always_on "),
    ]
    for key, label in order:
        print(_row(label, results[key]))

    # The key comparison: always-on with correction off vs decoupled with correction off
    drop_ao = results["always_on_corrected"]["success_rate"] - results["always_on_base"]["success_rate"]
    drop_dc = results["decoupled_triggered"]["success_rate"] - results["decoupled_base"]["success_rate"]
    print()
    print(f"  Co-adaptation gap  (always-on:  corrected − base)  = {drop_ao:+.1%}")
    print(f"  Residual gain      (decoupled:  triggered − base)  = {drop_dc:+.1%}")

    if drop_ao > 0.3 and drop_dc < 0.15:
        verdict = "CONFIRMS co-adaptation: always-on base collapses; decoupled base is independent."
    elif drop_ao > 0.15:
        verdict = "PARTIAL co-adaptation detected in always-on model."
    else:
        verdict = "Minimal co-adaptation gap — check if correction is actually active."
    print(f"\n  Verdict: {verdict}")
    print("=" * 75)

    if args.save_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_json)), exist_ok=True)
        with open(args.save_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.save_json}")

    eval_env.close()


if __name__ == "__main__":
    main()
