"""
Statistical Analysis: per-seed variance, Wilcoxon signed-rank, Cohen's d
Runs each seed independently to get per-seed accuracy for both models,
then computes proper statistics on the AMM advantage.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                 '..', 'sprint01'))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import json
from scipy import stats as scipy_stats

from tasks.variable_recall import VariableRecall
from models.adaptive_model_v3 import AdaptiveModelV3, FixedLSTM
from experiments.capacity_test import run


def cohens_d(a, b):
    """Cohen's d for paired samples."""
    diff = np.array(a) - np.array(b)
    return diff.mean() / diff.std(ddof=1)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--K",     nargs="+", type=int, default=[4, 8, 16])
    parser.add_argument("--hd",    nargs="+", type=int, default=[32, 128])
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--steps", type=int, default=5000)
    args = parser.parse_args()

    all_results = {}
    print("Collecting per-seed results...", flush=True)
    print(f"K={args.K}  hd={args.hd}  seeds={args.seeds}  steps={args.steps}\n", flush=True)

    for K in args.K:
        for hd in args.hd:
            key = f"K{K}_hd{hd}"
            lstm_accs = []
            amm_accs  = []
            for s in range(args.seeds):
                lstm_acc = run(K, hd, FixedLSTM,       max_steps=args.steps, seed=s)
                amm_acc  = run(K, hd, AdaptiveModelV3, max_steps=args.steps, seed=s)
                lstm_accs.append(lstm_acc)
                amm_accs.append(amm_acc)
                print(f"  K={K} hd={hd} seed={s}: LSTM={lstm_acc:.4f}  AMM={amm_acc:.4f}  delta={amm_acc-lstm_acc:+.4f}", flush=True)

            deltas = np.array(amm_accs) - np.array(lstm_accs)
            wilcox = scipy_stats.wilcoxon(amm_accs, lstm_accs, alternative='greater')
            cd     = cohens_d(amm_accs, lstm_accs)

            all_results[key] = {
                "K": K, "hd": hd,
                "lstm_accs": [round(x, 4) for x in lstm_accs],
                "amm_accs":  [round(x, 4) for x in amm_accs],
                "lstm_mean": round(float(np.mean(lstm_accs)), 4),
                "amm_mean":  round(float(np.mean(amm_accs)), 4),
                "delta_mean": round(float(deltas.mean()), 4),
                "delta_std":  round(float(deltas.std(ddof=1)), 4),
                "wilcoxon_p": round(float(wilcox.pvalue), 6),
                "cohens_d":   round(float(cd), 4),
            }
            print(f"  K={K} hd={hd} SUMMARY: delta={deltas.mean():+.4f} +/- {deltas.std(ddof=1):.4f}  "
                  f"p={wilcox.pvalue:.4f}  d={cd:.2f}\n", flush=True)

    # Summary table
    print("=" * 70, flush=True)
    print("STATISTICAL SUMMARY", flush=True)
    print("=" * 70, flush=True)
    print(f"{'Condition':<12} {'Delta mean':>12} {'Delta std':>10} {'p-value':>10} {'Cohen d':>9} {'Sig?':>6}", flush=True)
    print("-" * 70, flush=True)

    all_deltas = []
    all_sig = True
    for key, r in all_results.items():
        sig = "YES" if r["wilcoxon_p"] < 0.05 else "NO"
        if r["wilcoxon_p"] >= 0.05:
            all_sig = False
        all_deltas.append(r["delta_mean"])
        print(f"{key:<12} {r['delta_mean']:>+12.4f} {r['delta_std']:>10.4f} "
              f"{r['wilcoxon_p']:>10.4f} {r['cohens_d']:>9.2f} {sig:>6}", flush=True)

    print("-" * 70, flush=True)
    print(f"Overall mean delta: {np.mean(all_deltas):+.4f}", flush=True)
    cv = np.std(all_deltas) / abs(np.mean(all_deltas))
    print(f"CV across conditions: {cv:.3f}", flush=True)
    if all_sig:
        print("ALL CONDITIONS SIGNIFICANT (p < 0.05)", flush=True)
    else:
        print("WARNING: some conditions not significant", flush=True)

    os.makedirs("results", exist_ok=True)
    out_path = "results/stats_analysis.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFull per-seed results saved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
