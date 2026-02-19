"""
Full Adaptive AMM Experiment
-----------------------------
AMM with growth + pruning + merge enabled and in-loop stability monitoring.

Gate 3: dynamic growth is controlled (plateau, no explosion, no NaN).

Usage:
    python experiments/run_adaptive.py --seed 0
    python experiments/run_adaptive.py --seeds 5 --n 8
    python experiments/run_adaptive.py --n 4 8 16 32 --seeds 5
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
from tests.test_gates import gate_3_dynamic_growth_controlled
from tests.test_stability import (
    assert_no_nan_in_gradients,
    assert_gradient_norm_below,
    assert_loss_not_diverging,
    assert_slot_count_bounded,
    assert_slot_count_not_zero,
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_cfg(path: str = "config/default.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


class AdaptiveTrainerWithStabilityChecks(Trainer):
    """Trainer that runs in-loop stability assertions every eval period."""

    def train(self) -> dict:
        """Override to add in-loop stability monitoring."""
        import torch.nn as nn
        from collections import deque
        from training.metrics import (
            compute_grad_norm, has_nan_gradients, has_nan_in_output,
            compute_accuracy_from_logits
        )

        self.model.train()
        running_loss = 0.0
        running_acc  = 0.0
        epoch_steps  = 0
        epoch_write  = 0
        epoch_prune  = 0
        epoch_merge  = 0
        stability_violations = []

        if hasattr(self.model, "memory"):
            self.model.memory.reset_event_counts()

        for self.step in range(1, self.max_steps + 1):
            loss_val, acc_val, nan_detected = self._train_step()

            running_loss += loss_val if not torch.isnan(torch.tensor(loss_val)) else 0
            running_acc  += acc_val
            epoch_steps  += 1
            if nan_detected:
                self.nan_count += 1

            if hasattr(self.model, "memory"):
                epoch_write += self.model.memory.write_events
                epoch_prune += self.model.memory.prune_events
                epoch_merge += self.model.memory.merge_events
                self.model.memory.reset_event_counts()

            if self.step % self.log_every == 0:
                self.epoch += 1
                avg_loss = running_loss / max(epoch_steps, 1)
                avg_acc  = running_acc  / max(epoch_steps, 1)

                # --- In-loop stability checks ---
                try:
                    assert_gradient_norm_below(self.model, threshold=10.0)
                    assert_loss_not_diverging(list(self.loss_history), window=20)
                    if hasattr(self.model, "memory"):
                        assert_slot_count_bounded(
                            self.model.memory,
                            max_slots=self.model.memory.max_slots
                        )
                except AssertionError as e:
                    msg = f"[Stability] step={self.step}: {e}"
                    stability_violations.append(msg)
                    print(msg)

                active_slots = self._get_active_slots()
                avg_usage    = self._get_avg_usage()
                slot_entropy = self._get_slot_entropy()
                grad_norm    = compute_grad_norm(self.model)

                self.accuracy_history.append(avg_acc)
                self.slot_count_history.append(active_slots)

                record = {
                    "epoch":          self.epoch,
                    "step":           self.step,
                    "task_n":         self.task_n,
                    "loss":           round(avg_loss, 6),
                    "accuracy":       round(avg_acc, 6),
                    "active_slots":   active_slots,
                    "avg_slot_usage": round(avg_usage, 6),
                    "slot_entropy":   round(slot_entropy, 6),
                    "grad_norm":      round(grad_norm, 6),
                    "write_events":   epoch_write,
                    "prune_events":   epoch_prune,
                    "merge_events":   epoch_merge,
                    "nan_detected":   nan_detected,
                }
                self.logger.log(record)

                running_loss = 0.0
                running_acc  = 0.0
                epoch_steps  = 0
                epoch_write  = 0
                epoch_prune  = 0
                epoch_merge  = 0

        self.logger.close()
        self._build_results()
        self.results["stability_violations"] = stability_violations
        return self.results


def run_adaptive(n: int, seed: int, cfg: dict, results_dir: str, device: str) -> dict:
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

    exp_name = f"adaptive_amm_n{n}"
    trainer = AdaptiveTrainerWithStabilityChecks(
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",        type=int, default=None,
                        help="Single seed (overrides --seeds)")
    parser.add_argument("--seeds",       type=int, default=1)
    parser.add_argument("--n",           type=int, nargs="+", default=[8])
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--config",      default="config/default.yaml")
    parser.add_argument("--device",      default="cpu")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    gate_pass = True

    seeds = [args.seed] if args.seed is not None else list(range(args.seeds))

    for n in args.n:
        seed_results = []
        for seed in seeds:
            print(f"\n--- Adaptive AMM | N={n} | seed={seed} ---")
            r = run_adaptive(
                n=n, seed=seed, cfg=cfg,
                results_dir=args.results_dir, device=args.device,
            )
            seed_results.append(r)
            violations = r.get("stability_violations", [])
            print(f"    final_loss={r.get('final_loss', 'N/A'):.4f}  "
                  f"final_acc={r.get('final_acc', 0):.4f}  "
                  f"max_slots={r.get('slot_count_max', 0)}  "
                  f"stability_violations={len(violations)}")

        # Gate 3 check on each seed
        for r in seed_results:
            ok = gate_3_dynamic_growth_controlled(
                r,
                max_slots=cfg["memory"]["max_slots"],
                name=f"N={n} seed={r['seed']}",
            )
            gate_pass = gate_pass and ok

    print("\n=== ADAPTIVE AMM RUN COMPLETE ===")
    print("Gate 3:", "PASS" if gate_pass else "FAIL")
    return 0 if gate_pass else 1


if __name__ == "__main__":
    sys.exit(main())
