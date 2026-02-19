"""
Difficulty Ladder Experiments
------------------------------
Runs all models across all N values (4, 8, 16, 32) with 5 seeds each.
Produces a summary JSON and prints the performance delta matrix.

This is the Phase 4 experiment combining baselines + adaptive AMM across
the full difficulty ladder.

Usage:
    python experiments/run_difficulty.py
    python experiments/run_difficulty.py --n 4 8 16 32 --seeds 5
    python experiments/run_difficulty.py --models fixed_mlp ntm_lite adaptive
"""

import argparse
import json
import sys
import os
import random
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
from tasks.associative_recall import AssociativeRecall
from models.adaptive_model import FixedMLP, NTMLite, SmallTransformer, AdaptiveModel
from training.trainer import Trainer
from experiments.run_baseline import run_baseline_model
from experiments.run_adaptive import run_adaptive


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_cfg(path: str = "config/default.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def summarize_seeds(seed_results: list) -> dict:
    """Compute mean ± std across seeds."""
    accs  = [r.get("final_acc", 0.0)        for r in seed_results]
    slots = [r.get("slot_count_max", 0)     for r in seed_results]
    nans  = [r.get("nan_count", 0)          for r in seed_results]
    losses = [r.get("final_loss", 999)      for r in seed_results]
    return {
        "acc_mean":   float(np.mean(accs)),
        "acc_std":    float(np.std(accs)),
        "slot_mean":  float(np.mean(slots)),
        "slot_std":   float(np.std(slots)),
        "loss_mean":  float(np.mean(losses)),
        "loss_std":   float(np.std(losses)),
        "nan_total":  sum(nans),
        "n_seeds":    len(seed_results),
    }


def print_delta_matrix(summary: dict, n_values: list, adaptive_key: str = "adaptive") -> None:
    """Print accuracy delta: AMM - best_baseline for each N."""
    print("\n=== Performance Delta Matrix (AMM - Best Baseline) ===")
    print(f"{'N':>6} | {'Fixed MLP':>10} | {'NTM-lite':>10} | {'Transformer':>12} | {'AMM':>10} | {'Delta':>8}")
    print("-" * 70)
    for n in n_values:
        n_str = str(n)
        amm_acc = summary.get(adaptive_key, {}).get(n_str, {}).get("acc_mean", None)
        baseline_accs = {}
        for m in ["fixed_mlp", "ntm_lite", "transformer"]:
            baseline_accs[m] = summary.get(m, {}).get(n_str, {}).get("acc_mean", None)

        best_baseline = max((v for v in baseline_accs.values() if v is not None), default=None)
        delta = (amm_acc - best_baseline) if (amm_acc is not None and best_baseline is not None) else None

        def fmt(v):
            return f"{v:.3f}" if v is not None else "  N/A "

        print(f"{n:>6} | {fmt(baseline_accs.get('fixed_mlp')):>10} | "
              f"{fmt(baseline_accs.get('ntm_lite')):>10} | "
              f"{fmt(baseline_accs.get('transformer')):>12} | "
              f"{fmt(amm_acc):>10} | "
              f"{fmt(delta):>8}")


def print_slot_vs_n(summary: dict, n_values: list, adaptive_key: str = "adaptive") -> None:
    """Print slot count at convergence vs N."""
    print("\n=== Slot Count at Convergence vs N ===")
    print(f"{'N':>6} | {'Slots (mean)':>12} | {'Slots (std)':>11}")
    print("-" * 40)
    for n in n_values:
        n_str = str(n)
        s = summary.get(adaptive_key, {}).get(n_str, {})
        mean = s.get("slot_mean", None)
        std  = s.get("slot_std", None)
        if mean is not None:
            print(f"{n:>6} | {mean:>12.1f} | {std:>11.2f}")
        else:
            print(f"{n:>6} | {'N/A':>12} | {'N/A':>11}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",           type=int, nargs="+", default=[4, 8, 16, 32])
    parser.add_argument("--seeds",       type=int, default=5)
    parser.add_argument("--models",      nargs="+",
                        default=["fixed_mlp", "ntm_lite", "transformer", "adaptive"])
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--config",      default="config/default.yaml")
    parser.add_argument("--device",      default="cpu")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    summary = {}

    for model_name in args.models:
        summary[model_name] = {}
        for n in args.n:
            seed_results = []
            for seed in range(args.seeds):
                print(f"\n--- {model_name} | N={n} | seed={seed} ---")
                if model_name == "adaptive":
                    r = run_adaptive(
                        n=n, seed=seed, cfg=cfg,
                        results_dir=args.results_dir, device=args.device,
                    )
                else:
                    r = run_baseline_model(
                        model_name=model_name, n=n, seed=seed,
                        cfg=cfg, results_dir=args.results_dir, device=args.device,
                    )
                seed_results.append(r)
                print(f"    final_acc={r.get('final_acc', 0):.4f}  "
                      f"max_slots={r.get('slot_count_max', 0)}")

            summary[model_name][str(n)] = summarize_seeds(seed_results)

    # Print analysis tables
    print_delta_matrix(summary, args.n)
    print_slot_vs_n(summary, args.n)

    # Save summary
    os.makedirs(args.results_dir, exist_ok=True)
    summary_path = os.path.join(args.results_dir, "difficulty_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
