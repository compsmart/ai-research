"""
Statistical Analysis
---------------------
Wilcoxon signed-rank tests, Cohen's d, seed variance, performance delta matrix.
Prints a significance table and checks reproducibility.

Usage:
    python analysis/statistics.py results/
    python analysis/statistics.py results/ --alpha 0.05
"""

import argparse
import json
import sys
import os
import math
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.logger import Logger
from training.metrics import coefficient_of_variation, steps_to_threshold
from tests.test_gates import gate_reproducibility


# ------------------------------------------------------------------
# Statistical functions
# ------------------------------------------------------------------

def wilcoxon_signed_rank(x: list, y: list):
    """
    Simple Wilcoxon signed-rank test for paired samples x vs y.
    Returns (statistic, p_value). p_value is approximate.

    Falls back to scipy if available; otherwise uses a simplified calculation
    suitable for small samples (n <= 30).
    """
    try:
        from scipy.stats import wilcoxon
        if len(x) < 2 or len(y) < 2:
            return None, None
        stat, p = wilcoxon(x, y, alternative="two-sided", zero_method="wilcox")
        return float(stat), float(p)
    except ImportError:
        # Simplified: sign test as fallback when scipy unavailable
        n = min(len(x), len(y))
        diffs = [x[i] - y[i] for i in range(n)]
        nonzero = [d for d in diffs if d != 0]
        if not nonzero:
            return 0.0, 1.0
        pos = sum(1 for d in nonzero if d > 0)
        neg = len(nonzero) - pos
        # Approximate p-value via sign test binomial
        k = min(pos, neg)
        n_nz = len(nonzero)
        # Binomial probability P(X <= k) under H0: p=0.5
        p_approx = _binomial_tail(k, n_nz, 0.5)
        stat = pos - neg  # simplified
        return float(stat), min(1.0, 2 * p_approx)


def _binomial_tail(k: int, n: int, p: float) -> float:
    """P(X <= k) for X ~ Binomial(n, p)."""
    total = 0.0
    coeff = 1.0
    for i in range(k + 1):
        if i > 0:
            coeff *= (n - i + 1) / i
        total += coeff * (p ** i) * ((1 - p) ** (n - i))
    return total


def cohens_d(x: list, y: list) -> float:
    """Cohen's d effect size between two groups."""
    if len(x) < 2 or len(y) < 2:
        return 0.0
    nx, ny = len(x), len(y)
    mx = sum(x) / nx
    my = sum(y) / ny
    vx = sum((xi - mx)**2 for xi in x) / (nx - 1)
    vy = sum((yi - my)**2 for yi in y) / (ny - 1)
    pooled_sd = math.sqrt(((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2))
    if pooled_sd == 0:
        return 0.0
    return (mx - my) / pooled_sd


# ------------------------------------------------------------------
# Experiment data loading
# ------------------------------------------------------------------

def load_experiment_accs(results_dir: str, experiment_prefix: str,
                          n_values: list) -> dict:
    """
    Load final accuracy per seed for each N value.
    Returns {n: [acc_seed0, acc_seed1, ...]}
    """
    result = {}
    for n in n_values:
        exp_name = f"{experiment_prefix}_n{n}"
        seed_data = Logger.load_experiment(results_dir, exp_name)
        accs = []
        for seed, records in seed_data.items():
            if records:
                accs.append(records[-1].get("accuracy", 0.0))
        result[n] = accs
    return result


def load_summary_json(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


# ------------------------------------------------------------------
# Significance table
# ------------------------------------------------------------------

def print_significance_table(results_dir: str, n_values: list,
                               alpha: float = 0.05) -> None:
    """
    Compare AMM vs each baseline at each N.
    Prints statistic, p-value, Cohen's d, and verdict.
    """
    print("\n=== Statistical Significance Table ===")
    print(f"{'N':>4} | {'vs Model':>12} | {'W stat':>8} | {'p-value':>9} | "
          f"{'Cohen d':>8} | {'Verdict':>25}")
    print("-" * 85)

    amm_accs = load_experiment_accs(results_dir, "adaptive_amm", n_values)

    for n in n_values:
        amm = amm_accs.get(n, [])
        for model_prefix, label in [
            ("baseline_fixed_mlp",   "Fixed MLP"),
            ("baseline_ntm_lite",    "NTM-lite"),
            ("baseline_transformer", "Transformer"),
        ]:
            base_data = load_experiment_accs(results_dir, model_prefix, [n])
            base = base_data.get(n, [])

            if len(amm) < 2 or len(base) < 2:
                print(f"{n:>4} | {label:>12} | {'N/A':>8} | {'N/A':>9} | "
                      f"{'N/A':>8} | {'Insufficient data':>25}")
                continue

            stat, p = wilcoxon_signed_rank(amm, base)
            d = cohens_d(amm, base)
            amm_better = sum(amm) / len(amm) > sum(base) / len(base)

            if p is None:
                verdict = "No data"
            elif p < alpha and amm_better:
                verdict = f"AMM better (p<{alpha})"
            elif p < alpha and not amm_better:
                verdict = f"Baseline better (p<{alpha})"
            else:
                verdict = "No significant difference"

            p_str = f"{p:.4f}" if p is not None else "N/A"
            stat_str = f"{stat:.2f}" if stat is not None else "N/A"
            print(f"{n:>4} | {label:>12} | {stat_str:>8} | {p_str:>9} | "
                  f"{d:>8.3f} | {verdict:>25}")


# ------------------------------------------------------------------
# Reproducibility report
# ------------------------------------------------------------------

def print_reproducibility_report(results_dir: str, n_values: list) -> None:
    print("\n=== Reproducibility Report (CV across 5 seeds) ===")
    print(f"{'Experiment':>35} | {'N':>4} | {'CV':>8} | {'Pass':>5}")
    print("-" * 65)

    for exp_prefix, label in [
        ("adaptive_amm",          "AMM"),
        ("baseline_ntm_lite",     "NTM-lite"),
        ("baseline_fixed_mlp",    "Fixed MLP"),
        ("baseline_transformer",  "Transformer"),
    ]:
        for n in n_values:
            exp_name = f"{exp_prefix}_n{n}"
            seed_data = Logger.load_experiment(results_dir, exp_name)
            accs = []
            for seed, records in seed_data.items():
                if records:
                    accs.append(records[-1].get("accuracy", 0.0))

            if not accs:
                continue

            cv = coefficient_of_variation(accs)
            ok = cv < 0.10
            name = f"{label} N={n}"
            print(f"{name:>35} | {n:>4} | {cv:>8.4f} | {'YES' if ok else 'NO':>5}")


# ------------------------------------------------------------------
# Slot utilization entropy report
# ------------------------------------------------------------------

def print_entropy_report(results_dir: str, n_values: list) -> None:
    print("\n=== Slot Utilization Entropy (final epoch) ===")
    for n in n_values:
        exp_name = f"adaptive_amm_n{n}"
        seed_data = Logger.load_experiment(results_dir, exp_name)
        entropies = []
        for seed, records in seed_data.items():
            if records:
                entropies.append(records[-1].get("slot_entropy", 0.0))
        if entropies:
            mean_h = sum(entropies) / len(entropies)
            print(f"  N={n}: H = {mean_h:.4f} (avg over {len(entropies)} seeds)")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", nargs="?", default="results")
    parser.add_argument("--alpha",     type=float, default=0.05)
    parser.add_argument("--n",         type=int,   nargs="+", default=[4, 8, 16, 32])
    args = parser.parse_args()

    print(f"Results directory: {args.results_dir}")
    print(f"Significance level: alpha={args.alpha}")
    print(f"N values: {args.n}")

    print_significance_table(args.results_dir, args.n, alpha=args.alpha)
    print_reproducibility_report(args.results_dir, args.n)
    print_entropy_report(args.results_dir, args.n)

    # Load difficulty summary if available
    diff_path = os.path.join(args.results_dir, "difficulty_summary.json")
    summary = load_summary_json(diff_path)
    if summary:
        print("\n=== Difficulty Summary (from difficulty_summary.json) ===")
        for model_name, by_n in summary.items():
            print(f"\n  {model_name}:")
            for n_str, s in sorted(by_n.items(), key=lambda x: int(x[0])):
                print(f"    N={n_str}: acc={s.get('acc_mean', 0):.3f} "
                      f"+/- {s.get('acc_std', 0):.3f}  "
                      f"slots={s.get('slot_mean', 0):.1f} "
                      f"+/- {s.get('slot_std', 0):.1f}")

    # Formal hypothesis decision
    print("\n=== Hypothesis Decision Protocol ===")
    print("Hypothesis: AMM auto-calibrates capacity to task complexity.")
    print("Evidence required:")
    print("  - Slot count monotonically increases with N (visual check)")
    print("  - AMM >= NTM-lite accuracy at N>=16 with p<0.05")
    print("  - Reproducibility CV < 10% across 5 seeds")
    print("See test_gates.py for automated gate pass/fail.")


if __name__ == "__main__":
    main()
