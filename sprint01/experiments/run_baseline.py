"""
Baseline Experiments
---------------------
Trains Fixed MLP, NTM-lite (20 slots), and Small Transformer baselines.
5 seeds each, all difficulty levels.

Usage:
    python experiments/run_baseline.py
    python experiments/run_baseline.py --seeds 5 --n 8
    python experiments/run_baseline.py --n 4 8 16 32 --seeds 5
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
from models.adaptive_model import FixedMLP, FixedLSTM, NTMLite, SmallTransformer
from training.trainer import Trainer
from tests.test_gates import gate_1_baseline_converges


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_cfg(path: str = "config/default.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run_baseline_model(model_name: str, n: int, seed: int, cfg: dict,
                        results_dir: str, device: str) -> dict:
    set_seed(seed)
    task = AssociativeRecall(
        n=n,
        vocab_size=max(10, n),
        key_dim=cfg["associative_recall"]["key_dim"],
        val_dim=cfg["associative_recall"]["val_dim"],
        device=device,
    )

    if model_name == "fixed_mlp":
        seq_len = n + 1
        model = FixedMLP(
            input_dim=task.input_dim,
            seq_len=seq_len,
            output_dim=task.output_dim,
            hidden_dim=cfg["model"]["hidden_dim"],
            n_layers=cfg["model"]["n_layers"],
        )
    elif model_name == "fixed_lstm":
        model = FixedLSTM(
            input_dim=task.input_dim,
            output_dim=task.output_dim,
            hidden_dim=cfg["model"]["hidden_dim"],
        )
    elif model_name == "ntm_lite":
        model = NTMLite(
            input_dim=task.input_dim,
            output_dim=task.output_dim,
            n_slots=cfg["ntm_lite"]["n_slots"],
            hidden_dim=cfg["model"]["hidden_dim"],
            d_key=cfg["ntm_lite"]["d_key"],
            d_val=cfg["ntm_lite"]["d_val"],
            temp=cfg["ntm_lite"]["temp"],
            encoder=cfg["model"].get("encoder", "lstm"),
        )
    elif model_name == "transformer":
        model = SmallTransformer(
            input_dim=task.input_dim,
            output_dim=task.output_dim,
            d_model=cfg["transformer"]["d_model"],
            n_heads=cfg["transformer"]["n_heads"],
            n_layers=cfg["transformer"]["n_layers"],
            d_ff=cfg["transformer"]["d_ff"],
            dropout=cfg["transformer"]["dropout"],
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    exp_name = f"baseline_{model_name}_n{n}"
    trainer = Trainer(
        model=model,
        task=task,
        cfg=cfg,
        experiment_name=exp_name,
        seed=seed,
        results_dir=results_dir,
        task_n=n,
        device=device,
    )
    results = trainer.train()
    results["model"] = model_name
    results["n"] = n
    results["seed"] = seed
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds",      type=int, default=5)
    parser.add_argument("--n",          type=int, nargs="+", default=[4, 8, 16, 32])
    parser.add_argument("--models",     nargs="+",
                        default=["fixed_lstm", "ntm_lite", "transformer"])
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--config",     default="config/default.yaml")
    parser.add_argument("--device",     default="cpu")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    all_results = {}
    gate_pass = True

    for model_name in args.models:
        all_results[model_name] = {}
        for n in args.n:
            seed_results = []
            for seed in range(args.seeds):
                print(f"\n--- Baseline: {model_name} | N={n} | seed={seed} ---")
                r = run_baseline_model(
                    model_name=model_name, n=n, seed=seed,
                    cfg=cfg, results_dir=args.results_dir, device=args.device,
                )
                seed_results.append(r)
                print(f"    final_loss={r.get('final_loss', 'N/A'):.4f}  "
                      f"final_acc={r.get('final_acc', 0):.4f}")

            all_results[model_name][n] = seed_results

            # Gate 1 check on mean results
            mean_result = {
                "final_loss":    sum(r.get("final_loss", 999) for r in seed_results) / len(seed_results),
                "nan_count":     sum(r.get("nan_count", 1) for r in seed_results),
                "grad_norm_max": max(r.get("grad_norm_max", 0) for r in seed_results),
            }
            ok = gate_1_baseline_converges(mean_result, name=f"{model_name} N={n}")
            gate_pass = gate_pass and ok

    # Save summary
    summary_path = os.path.join(args.results_dir, "baseline_summary.json")
    os.makedirs(args.results_dir, exist_ok=True)
    with open(summary_path, "w") as f:
        # Convert to serializable format
        serializable = {}
        for m, by_n in all_results.items():
            serializable[m] = {}
            for n, seed_list in by_n.items():
                serializable[m][str(n)] = [
                    {k: (v if not isinstance(v, list) else v[-5:])
                     for k, v in r.items()}
                    for r in seed_list
                ]
        json.dump(serializable, f, indent=2)

    print(f"\nSummary saved to {summary_path}")
    print("\n=== BASELINE RUN COMPLETE ===")
    print("Gate 1:", "PASS" if gate_pass else "FAIL")
    return 0 if gate_pass else 1


if __name__ == "__main__":
    sys.exit(main())
