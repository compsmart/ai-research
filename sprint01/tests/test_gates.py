"""
Automated Gate Evaluation (Phases 1 / 2 / 3)
---------------------------------------------
Run with:
    python tests/test_gates.py --gate 1
    python tests/test_gates.py --gate 2
    python tests/test_gates.py --gate 3
    python tests/test_gates.py --gate 1 --gate 2 --gate 3

Also importable as functions for use inside experiment scripts.
"""

import argparse
import json
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.logger import Logger
from training.metrics import coefficient_of_variation


# ------------------------------------------------------------------
# Gate functions
# ------------------------------------------------------------------

def gate_1_baseline_converges(results: dict, name: str = "") -> bool:
    """
    Gate 1: baseline trained successfully.
    - final_loss < 0.1
    - grad_norm_max < 10.0
    - nan_count == 0
    """
    ok = True
    prefix = f"[Gate 1{' '+name if name else ''}]"

    if results.get("nan_count", 1) > 0:
        print(f"{prefix} FAIL: nan_count={results['nan_count']} (expected 0)")
        ok = False
    if results.get("final_loss", 999) >= 0.1:
        print(f"{prefix} FAIL: final_loss={results.get('final_loss', 'N/A'):.4f} (expected < 0.1)")
        ok = False
    if results.get("grad_norm_max", 999) >= 10.0:
        print(f"{prefix} FAIL: grad_norm_max={results.get('grad_norm_max', 'N/A'):.4f} (expected < 10.0)")
        ok = False

    if ok:
        print(f"{prefix} PASS — loss={results.get('final_loss','N/A'):.4f}, "
              f"nan={results.get('nan_count',0)}")
    return ok


def gate_2_static_memory_stable(static_results: dict, baseline_results: dict,
                                  name: str = "") -> bool:
    """
    Gate 2: static memory must not be >5% worse than best baseline accuracy.
    - static_acc >= baseline_acc * 0.95
    - nan_count == 0
    """
    ok = True
    prefix = f"[Gate 2{' '+name if name else ''}]"

    static_acc   = static_results.get("final_acc", 0.0)
    baseline_acc = baseline_results.get("final_acc", 0.0)

    if static_results.get("nan_count", 1) > 0:
        print(f"{prefix} FAIL: nan_count={static_results['nan_count']}")
        ok = False
    if static_acc < baseline_acc * 0.95:
        print(f"{prefix} FAIL: static_acc={static_acc:.4f} < "
              f"95% of baseline_acc={baseline_acc:.4f} ({baseline_acc * 0.95:.4f})")
        ok = False

    if ok:
        print(f"{prefix} PASS — static_acc={static_acc:.4f}, "
              f"baseline_acc={baseline_acc:.4f}")
    return ok


def gate_3_dynamic_growth_controlled(results: dict, max_slots: int = 100,
                                      name: str = "") -> bool:
    """
    Gate 3: adaptive growth is controlled.
    - nan_count == 0
    - slot_count_max < max_slots
    - last 20% of training shows <5% slot change (plateau)
    """
    ok = True
    prefix = f"[Gate 3{' '+name if name else ''}]"

    if results.get("nan_count", 1) > 0:
        print(f"{prefix} FAIL: nan_count={results['nan_count']}")
        ok = False

    slot_max = results.get("slot_count_max", max_slots + 1)
    if slot_max >= max_slots:
        print(f"{prefix} FAIL: slot_count_max={slot_max} >= max_slots={max_slots}")
        ok = False

    # Check plateau in last 20% of training
    slot_counts = results.get("slot_counts", [])
    if slot_counts:
        late_start = int(0.8 * len(slot_counts))
        late = slot_counts[late_start:]
        if late and max(late) > 0:
            change = (max(late) - min(late)) / max(late)
            if change >= 0.05:
                print(f"{prefix} FAIL: slot change in last 20% = {change:.3f} >= 0.05")
                ok = False
        else:
            print(f"{prefix} WARN: no slot data in last 20%")
    else:
        print(f"{prefix} WARN: no slot_counts recorded")

    if ok:
        print(f"{prefix} PASS — max_slots={slot_max}, "
              f"late_change={results.get('late_slot_change', 0.0):.3f}")
    return ok


def gate_reproducibility(results_by_seed: list, name: str = "") -> bool:
    """
    Reproducibility: coefficient of variation of final accuracy < 10%.
    """
    prefix = f"[Reproducibility{' '+name if name else ''}]"
    accs = [r["final_acc"] for r in results_by_seed if "final_acc" in r]
    if not accs:
        print(f"{prefix} WARN: no accuracy data")
        return True
    cv = coefficient_of_variation(accs)
    if cv >= 0.10:
        print(f"{prefix} FAIL: CV={cv:.4f} >= 0.10 (accs={accs})")
        return False
    print(f"{prefix} PASS — CV={cv:.4f}, accs={[f'{a:.3f}' for a in accs]}")
    return True


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def load_latest_results(results_dir: str, experiment: str) -> list:
    """Load final_acc and related fields from all seed logs of an experiment."""
    seed_data = Logger.load_experiment(results_dir, experiment)
    summaries = []
    for seed, records in seed_data.items():
        if not records:
            continue
        last = records[-1]
        summaries.append({
            "seed": seed,
            "final_loss":    last.get("loss", 999),
            "final_acc":     last.get("accuracy", 0.0),
            "nan_count":     sum(1 for r in records if r.get("nan_detected", False)),
            "grad_norm_max": max((r.get("grad_norm", 0) for r in records), default=0),
            "slot_count_max": max((r.get("active_slots", 0) for r in records), default=0),
            "slot_counts":   [r.get("active_slots", 0) for r in records],
            "late_slot_change": _compute_late_change(
                [r.get("active_slots", 0) for r in records]
            ),
        })
    return summaries


def _compute_late_change(slot_counts: list) -> float:
    if not slot_counts:
        return 0.0
    n = len(slot_counts)
    late = slot_counts[int(0.8 * n):]
    if not late or max(late) == 0:
        return 0.0
    return (max(late) - min(late)) / max(late)


def main():
    parser = argparse.ArgumentParser(description="Run AMM gate evaluations")
    parser.add_argument("--gate", type=int, action="append", default=[],
                        choices=[1, 2, 3], help="Which gate(s) to evaluate")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--baseline-exp",     default="baseline_ntm_lite")
    parser.add_argument("--static-exp",       default="static_memory")
    parser.add_argument("--adaptive-exp",     default="adaptive_amm")
    parser.add_argument("--max-slots",        type=int, default=100)
    args = parser.parse_args()

    gates = args.gate or [1, 2, 3]
    all_pass = True

    if 1 in gates:
        print("\n=== Gate 1: Baseline Convergence ===")
        results = load_latest_results(args.results_dir, args.baseline_exp)
        if not results:
            print(f"[Gate 1] WARN: No results found for '{args.baseline_exp}'")
        else:
            # Use mean across seeds
            mean_result = {
                "final_loss":    sum(r["final_loss"] for r in results) / len(results),
                "nan_count":     sum(r["nan_count"] for r in results),
                "grad_norm_max": max(r["grad_norm_max"] for r in results),
            }
            ok = gate_1_baseline_converges(mean_result)
            all_pass = all_pass and ok

    if 2 in gates:
        print("\n=== Gate 2: Static Memory Stability ===")
        static_results   = load_latest_results(args.results_dir, args.static_exp)
        baseline_results = load_latest_results(args.results_dir, args.baseline_exp)
        if not static_results or not baseline_results:
            print("[Gate 2] WARN: Missing results — skipping")
        else:
            def mean_r(rs):
                return {
                    "final_acc": sum(r["final_acc"] for r in rs) / len(rs),
                    "nan_count": sum(r["nan_count"] for r in rs),
                }
            ok = gate_2_static_memory_stable(mean_r(static_results), mean_r(baseline_results))
            all_pass = all_pass and ok

    if 3 in gates:
        print("\n=== Gate 3: Dynamic Growth Controlled ===")
        adaptive_results = load_latest_results(args.results_dir, args.adaptive_exp)
        if not adaptive_results:
            print(f"[Gate 3] WARN: No results found for '{args.adaptive_exp}'")
        else:
            # Use worst-case seed for gate 3
            for r in adaptive_results:
                ok = gate_3_dynamic_growth_controlled(r, max_slots=args.max_slots,
                                                       name=f"seed={r['seed']}")
                all_pass = all_pass and ok

    print("\n" + ("=== ALL GATES PASSED ===" if all_pass else "=== SOME GATES FAILED ==="))
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
