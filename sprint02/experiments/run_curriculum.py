"""
Curriculum Recall Experiments
-------------------------------
Trains AdaptiveModelV2, NTMLite, and FixedLSTM on the CurriculumRecall task
across the K difficulty ladder (4, 8, 16, 32), 5 seeds each.

Key measurements:
  - Final accuracy per model per K
  - Slot count at convergence vs K (the main adaptive hypothesis)
  - Novelty ratio: does the trigger fire on genuine novelty?
  - Gate 3: does slot count plateau (no oscillation)?

Usage:
    cd sprint02/
    python experiments/run_curriculum.py
    python experiments/run_curriculum.py --K 4 8 16 32 --seeds 5
    python experiments/run_curriculum.py --models adaptive ntm_lite fixed_lstm --K 8
"""

import argparse, json, os, sys, random
import numpy as np
import torch

_sprint02_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_sprint01_dir = os.path.join(_sprint02_dir, '..', 'sprint01')
# sprint01 inserted first so sprint02 at position 0 overrides it
sys.path.insert(0, _sprint01_dir)
sys.path.insert(0, _sprint02_dir)

import yaml
from tasks.curriculum_recall import CurriculumRecall
from models.adaptive_model_v2 import AdaptiveModelV2, FixedLSTM, NTMLite
from training.trainer_v2 import TrainerV2


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def load_cfg(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def n_shown_for_K(K: int, mode: str) -> int:
    if mode == "half":
        return K // 2
    elif mode == "zero":
        return 1          # minimum (n_shown must be >= 1 to avoid degenerate seq)
    else:
        return int(mode)


def build_model(model_name: str, task: CurriculumRecall, cfg: dict) -> torch.nn.Module:
    mem    = cfg["memory"]
    ntm    = cfg["ntm_lite"]
    hidden = cfg["model"]["hidden_dim"]
    cr_cfg = cfg["curriculum_recall"]

    if model_name == "adaptive":
        return AdaptiveModelV2(
            input_dim=task.input_dim,
            output_dim=task.output_dim,
            hidden_dim=hidden,
            key_dim=cr_cfg["key_dim"],   # concept key dim for memory addressing
            max_slots=mem["max_slots"],
            d_key=mem["d_key"],
            d_val=mem["d_val"],
            temp=mem["temp"],
            novelty_threshold=mem["novelty_threshold"],
            loss_floor=mem["loss_floor"],
            usage_ema_decay=mem["usage_ema_decay"],
            min_usage=mem["min_usage"],
            min_age=mem["min_age"],
            prune_every=mem["prune_every"],
            merge_threshold=mem["merge_threshold"],
            merge_every=mem["merge_every"],
            write_lr=mem["write_lr"],
        )
    elif model_name == "ntm_lite":
        return NTMLite(
            input_dim=task.input_dim,
            output_dim=task.output_dim,
            n_slots=ntm["n_slots"],
            hidden_dim=hidden,
            d_key=ntm["d_key"],
            d_val=ntm["d_val"],
            temp=ntm["temp"],
            encoder="lstm",
        )
    elif model_name == "fixed_lstm":
        return FixedLSTM(
            input_dim=task.input_dim,
            output_dim=task.output_dim,
            hidden_dim=hidden,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")


def run_one(model_name: str, K: int, seed: int, cfg: dict,
            results_dir: str, device: str) -> dict:
    set_seed(seed)
    cr_cfg = cfg["curriculum_recall"]
    n_shown = n_shown_for_K(K, cfg["experiment"].get("n_shown_mode", "half"))

    task = CurriculumRecall(
        K=K,
        n_shown=n_shown,
        vocab_size=cr_cfg["vocab_size"],
        key_dim=cr_cfg["key_dim"],
        val_dim=cr_cfg["val_dim"],
        concept_seed=seed,   # each seed gets its own concept dict
        device=device,
    )

    model = build_model(model_name, task, cfg)
    exp_name = f"curriculum_{model_name}_K{K}"

    trainer = TrainerV2(
        model=model,
        task=task,
        cfg=cfg,
        experiment_name=exp_name,
        seed=seed,
        results_dir=results_dir,
        task_K=K,
        device=device,
    )
    results = trainer.train()
    results.update({"model": model_name, "K": K, "seed": seed,
                    "n_shown": n_shown, "memory_query_rate": task.memory_query_rate})
    return results


def summarize(seed_results: list) -> dict:
    accs   = [r.get("final_acc", 0)       for r in seed_results]
    slots  = [r.get("slot_count_max", 0)  for r in seed_results]
    nans   = [r.get("nan_count", 0)       for r in seed_results]
    losses = [r.get("final_loss", 999)    for r in seed_results]
    return {
        "acc_mean":  float(np.mean(accs)),
        "acc_std":   float(np.std(accs)),
        "slot_mean": float(np.mean(slots)),
        "slot_std":  float(np.std(slots)),
        "loss_mean": float(np.mean(losses)),
        "nan_total": sum(nans),
        "n_seeds":   len(seed_results),
    }


def print_results(summary: dict, K_values: list) -> None:
    models = list(summary.keys())
    print(f"\n{'Model':>12}  " + "  ".join(f"K={k:>2}" for k in K_values))
    print("-" * (14 + 10 * len(K_values)))
    for m in models:
        row = f"{m:>12}  "
        for K in K_values:
            s = summary[m].get(str(K), {})
            row += f"{s.get('acc_mean', 0):>8.4f}  "
        print(row)

    print(f"\n=== Slot count at convergence vs K (adaptive) ===")
    print(f"{'K':>4}  {'slot_mean':>10}  {'slot_std':>9}")
    for K in K_values:
        s = summary.get("adaptive", {}).get(str(K), {})
        if s:
            print(f"{K:>4}  {s['slot_mean']:>10.1f}  {s['slot_std']:>9.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--K",           type=int, nargs="+", default=[4, 8, 16, 32])
    parser.add_argument("--seeds",       type=int, default=5)
    parser.add_argument("--models",      nargs="+",
                        default=["fixed_lstm", "ntm_lite", "adaptive"])
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--config",      default="config/sprint02.yaml")
    parser.add_argument("--device",      default="cpu")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    os.makedirs(args.results_dir, exist_ok=True)
    summary = {}

    for model_name in args.models:
        summary[model_name] = {}
        for K in args.K:
            seed_results = []
            for seed in range(args.seeds):
                print(f"\n--- {model_name} | K={K} | seed={seed} ---")
                r = run_one(model_name, K, seed, cfg, args.results_dir, args.device)
                seed_results.append(r)
                slot_str = (f"  slots={r.get('slot_count_max',0)}"
                            if model_name == "adaptive" else "")
                print(f"    acc={r.get('final_acc',0):.4f}  "
                      f"loss={r.get('final_loss',0):.4f}{slot_str}")
            summary[model_name][str(K)] = summarize(seed_results)

    print_results(summary, args.K)

    summary_path = os.path.join(args.results_dir, "curriculum_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary -> {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
