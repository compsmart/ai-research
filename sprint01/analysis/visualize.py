"""
Visualization
--------------
Loss curves, memory growth, accuracy vs slot count, difficulty ladder plots.

Usage:
    python analysis/visualize.py results/
    python analysis/visualize.py results/ --output figures/
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.logger import Logger


def try_import_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        print("[visualize] matplotlib not installed. Text-only output.")
        return None


# ------------------------------------------------------------------
# Text-based fallback visualizations
# ------------------------------------------------------------------

def print_loss_curve(records: list, title: str = "", width: int = 60) -> None:
    """ASCII sparkline of loss over epochs."""
    losses = [r.get("loss", 0) for r in records if "loss" in r]
    if not losses:
        print(f"[{title}] No loss data")
        return
    lo, hi = min(losses), max(losses)
    blocks = " .:-=+*#@"
    bar = ""
    step = max(1, len(losses) // width)
    for i in range(0, len(losses), step):
        v = losses[i]
        idx = int((v - lo) / (hi - lo + 1e-8) * (len(blocks) - 1))
        bar += blocks[idx]
    print(f"[{title}] Loss curve ({len(losses)} epochs)")
    print(f"  hi={hi:.4f} [{bar}] lo={lo:.4f}")


def print_slot_curve(records: list, title: str = "", width: int = 60) -> None:
    """ASCII display of active slot count over epochs."""
    slots = [r.get("active_slots", 0) for r in records if "active_slots" in r]
    if not slots:
        return
    lo, hi = min(slots), max(slots)
    blocks = " .:-=+*#@"
    bar = ""
    step = max(1, len(slots) // width)
    for i in range(0, len(slots), step):
        v = slots[i]
        idx = int((v - lo) / (hi - lo + 1e-8) * (len(blocks) - 1))
        bar += blocks[idx]
    print(f"[{title}] Slot count ({lo} to {hi})")
    print(f"  [{bar}]")


# ------------------------------------------------------------------
# Matplotlib plots (if available)
# ------------------------------------------------------------------

def plot_learning_curves(all_data: dict, output_dir: str, plt) -> None:
    """Plot loss curves for all experiments on same axes."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for exp_name, seed_dict in all_data.items():
        for seed, records in seed_dict.items():
            epochs = [r["epoch"] for r in records]
            losses = [r.get("loss", None) for r in records]
            accs   = [r.get("accuracy", None) for r in records]
            label  = f"{exp_name[:20]} s{seed}"
            axes[0].plot(epochs, losses, alpha=0.6, linewidth=0.8, label=label)
            axes[1].plot(epochs, accs,   alpha=0.6, linewidth=0.8, label=label)

    axes[0].set_title("Loss over Training")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend(fontsize=6, ncol=2)

    axes[1].set_title("Accuracy over Training")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend(fontsize=6, ncol=2)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "learning_curves.png"), dpi=150)
    plt.close()
    print(f"Saved: {output_dir}/learning_curves.png")


def plot_slot_growth(adaptive_data: dict, output_dir: str, plt) -> None:
    """Plot active slot count over training for adaptive experiments."""
    fig, ax = plt.subplots(figsize=(10, 5))

    for exp_name, seed_dict in adaptive_data.items():
        if "adaptive" not in exp_name:
            continue
        for seed, records in seed_dict.items():
            epochs = [r["epoch"] for r in records]
            slots  = [r.get("active_slots", 0) for r in records]
            ax.plot(epochs, slots, alpha=0.7, label=f"{exp_name[:20]} s{seed}")

    ax.set_title("Memory Slot Growth over Training (Adaptive AMM)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Active Slots")
    ax.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "slot_growth.png"), dpi=150)
    plt.close()
    print(f"Saved: {output_dir}/slot_growth.png")


def plot_slot_vs_n(difficulty_summary_path: str, output_dir: str, plt) -> None:
    """Plot slot count at convergence vs N (difficulty)."""
    if not os.path.exists(difficulty_summary_path):
        print(f"[plot_slot_vs_n] File not found: {difficulty_summary_path}")
        return

    with open(difficulty_summary_path) as f:
        summary = json.load(f)

    fig, ax = plt.subplots(figsize=(7, 5))

    if "adaptive" in summary:
        ns    = sorted(int(n) for n in summary["adaptive"].keys())
        means = [summary["adaptive"][str(n)].get("slot_mean", 0) for n in ns]
        stds  = [summary["adaptive"][str(n)].get("slot_std", 0)  for n in ns]
        ax.errorbar(ns, means, yerr=stds, marker="o", label="AMM (adaptive)",
                    linewidth=2, capsize=4)

    ax.set_title("Active Slot Count at Convergence vs Task Difficulty (N)")
    ax.set_xlabel("N (key-value pairs)")
    ax.set_ylabel("Active Slots at Convergence")
    ax.set_xticks([4, 8, 16, 32])
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "slot_vs_n.png"), dpi=150)
    plt.close()
    print(f"Saved: {output_dir}/slot_vs_n.png")


def plot_accuracy_vs_n(difficulty_summary_path: str, output_dir: str, plt) -> None:
    """Plot final accuracy for all models vs N."""
    if not os.path.exists(difficulty_summary_path):
        return

    with open(difficulty_summary_path) as f:
        summary = json.load(f)

    fig, ax = plt.subplots(figsize=(9, 5))
    model_styles = {
        "fixed_mlp":   ("Fixed MLP",      "s--",  "gray"),
        "ntm_lite":    ("NTM-lite",        "^--",  "blue"),
        "transformer": ("Transformer",     "D--",  "orange"),
        "adaptive":    ("AMM (adaptive)",  "o-",   "red"),
    }

    for key, (label, style, color) in model_styles.items():
        if key not in summary:
            continue
        ns    = sorted(int(n) for n in summary[key].keys())
        means = [summary[key][str(n)].get("acc_mean", 0) for n in ns]
        stds  = [summary[key][str(n)].get("acc_std", 0)  for n in ns]
        ax.errorbar(ns, means, yerr=stds, fmt=style, color=color,
                    label=label, linewidth=2, capsize=4, markersize=6)

    ax.set_title("Final Accuracy vs Task Difficulty (N)")
    ax.set_xlabel("N (key-value pairs)")
    ax.set_ylabel("Accuracy")
    ax.set_xticks([4, 8, 16, 32])
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "accuracy_vs_n.png"), dpi=150)
    plt.close()
    print(f"Saved: {output_dir}/accuracy_vs_n.png")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir",  nargs="?", default="results")
    parser.add_argument("--output",     default=None,
                        help="Output directory for figures (default: results_dir/figures)")
    args = parser.parse_args()

    output_dir = args.output or os.path.join(args.results_dir, "figures")
    plt = try_import_matplotlib()

    # Load all experiment logs
    results_base = Path(args.results_dir)
    all_data = {}
    if results_base.exists():
        for exp_dir in sorted(results_base.iterdir()):
            if exp_dir.is_dir() and not exp_dir.name.startswith("."):
                exp_data = Logger.load_experiment(args.results_dir, exp_dir.name)
                if exp_data:
                    all_data[exp_dir.name] = exp_data

    if not all_data:
        print("[visualize] No experiment data found in", args.results_dir)
        return

    # Text summaries (always)
    for exp_name, seed_dict in all_data.items():
        for seed, records in seed_dict.items():
            print_loss_curve(records, title=f"{exp_name}/seed={seed}")
            print_slot_curve(records, title=f"{exp_name}/seed={seed}")

    # Matplotlib plots (if available)
    if plt:
        plot_learning_curves(all_data, output_dir, plt)
        plot_slot_growth(all_data, output_dir, plt)

        diff_summary = os.path.join(args.results_dir, "difficulty_summary.json")
        plot_slot_vs_n(diff_summary, output_dir, plt)
        plot_accuracy_vs_n(diff_summary, output_dir, plt)

        print(f"\nAll figures saved to: {output_dir}")
    else:
        print("\nInstall matplotlib for graphical plots: pip install matplotlib")


if __name__ == "__main__":
    main()
