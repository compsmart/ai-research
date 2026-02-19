"""
Static Memory Experiment
-------------------------
AMM with growth disabled (fixed 20 active slots).
Used to verify memory mechanism correctness before enabling adaptive growth.

Gate 2: static_acc >= baseline_acc * 0.95

Usage:
    python experiments/run_static_memory.py
    python experiments/run_static_memory.py --seeds 5 --n 8
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
from models.adaptive_model import AdaptiveModel
from training.trainer import Trainer
from training.logger import Logger
from tests.test_gates import gate_2_static_memory_stable


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_cfg(path: str = "config/default.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run_static_memory(n: int, seed: int, cfg: dict,
                       results_dir: str, device: str) -> dict:
    set_seed(seed)
    task = AssociativeRecall(
        n=n,
        vocab_size=max(10, n),
        key_dim=cfg["associative_recall"]["key_dim"],
        val_dim=cfg["associative_recall"]["val_dim"],
        device=device,
    )

    model = AdaptiveModel(
        input_dim=task.input_dim,
        output_dim=task.output_dim,
        hidden_dim=cfg["model"]["hidden_dim"],
        n_layers=cfg["model"]["n_layers"],
        encoder=cfg["model"].get("encoder", "lstm"),
        max_slots=cfg["memory"]["max_slots"],
        d_key=cfg["memory"]["d_key"],
        d_val=cfg["memory"]["d_val"],
        temp=cfg["memory"]["temp"],
        error_threshold=cfg["memory"]["error_threshold"],
        usage_ema_decay=cfg["memory"]["usage_ema_decay"],
        min_usage=cfg["memory"]["min_usage"],
        min_age=cfg["memory"]["min_age"],
        prune_every=cfg["memory"]["prune_every"],
        merge_threshold=cfg["memory"]["merge_threshold"],
        merge_every=cfg["memory"]["merge_every"],
        write_lr=cfg["memory"]["write_lr"],
    )

    # Disable growth; pre-activate 20 slots
    model.memory.growth_disabled = True
    n_init_slots = cfg["ntm_lite"]["n_slots"]  # 20
    with torch.no_grad():
        model.memory.active_mask[:n_init_slots] = True
        torch.nn.init.normal_(model.memory.slots_key[:n_init_slots], std=0.1)
        torch.nn.init.normal_(model.memory.slots_value[:n_init_slots], std=0.1)

    exp_name = f"static_memory_n{n}"
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
    results["n"] = n
    results["seed"] = seed
    return results


def load_baseline_results(results_dir: str, n: int) -> dict:
    """Load best baseline final_acc for gate 2 comparison."""
    best_acc = 0.0
    best_nan = 0
    for model_name in ["ntm_lite", "transformer", "fixed_mlp"]:
        exp_name = f"baseline_{model_name}_n{n}"
        seed_data = Logger.load_experiment(results_dir, exp_name)
        for seed, records in seed_data.items():
            if records:
                acc = records[-1].get("accuracy", 0.0)
                best_acc = max(best_acc, acc)
    return {"final_acc": best_acc, "nan_count": 0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds",       type=int, default=5)
    parser.add_argument("--n",           type=int, nargs="+", default=[4, 8, 16, 32])
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--config",      default="config/default.yaml")
    parser.add_argument("--device",      default="cpu")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    gate_pass = True

    for n in args.n:
        seed_results = []
        for seed in range(args.seeds):
            print(f"\n--- Static Memory | N={n} | seed={seed} ---")
            r = run_static_memory(
                n=n, seed=seed, cfg=cfg,
                results_dir=args.results_dir, device=args.device,
            )
            seed_results.append(r)
            print(f"    final_loss={r.get('final_loss', 'N/A'):.4f}  "
                  f"final_acc={r.get('final_acc', 0):.4f}  "
                  f"active_slots={r.get('slot_count_max', 0)}")

        # Gate 2 check
        mean_static = {
            "final_acc": sum(r.get("final_acc", 0) for r in seed_results) / len(seed_results),
            "nan_count": sum(r.get("nan_count", 0) for r in seed_results),
        }
        baseline = load_baseline_results(args.results_dir, n)

        if baseline["final_acc"] == 0.0:
            print(f"  [Gate 2] WARN: no baseline results found for N={n}, skipping gate check")
        else:
            ok = gate_2_static_memory_stable(mean_static, baseline, name=f"N={n}")
            gate_pass = gate_pass and ok

    print("\n=== STATIC MEMORY RUN COMPLETE ===")
    print("Gate 2:", "PASS" if gate_pass else "FAIL")
    return 0 if gate_pass else 1


if __name__ == "__main__":
    sys.exit(main())
