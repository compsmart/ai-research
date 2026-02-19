"""
5-Seed Sweep Wrapper
---------------------
Runs a target experiment script across all 5 seeds sequentially.

Usage:
    python experiments/run_all_seeds.py --experiment adaptive --n 8
    python experiments/run_all_seeds.py --experiment baseline --n 4 8 16 32
    python experiments/run_all_seeds.py --experiment all --n 4 8 16 32
"""

import argparse
import sys
import os
import subprocess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


EXPERIMENT_SCRIPTS = {
    "baseline":  "experiments/run_baseline.py",
    "static":    "experiments/run_static_memory.py",
    "adaptive":  "experiments/run_adaptive.py",
    "difficulty": "experiments/run_difficulty.py",
}


def run_script(script: str, n_values: list, seeds: int, results_dir: str,
               config: str, device: str, extra_args: list = None) -> int:
    cmd = [
        sys.executable, script,
        "--seeds", str(seeds),
        "--n", *[str(n) for n in n_values],
        "--results-dir", results_dir,
        "--config", config,
        "--device", device,
    ]
    if extra_args:
        cmd.extend(extra_args)
    print(f"\nRunning: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=list(EXPERIMENT_SCRIPTS.keys()) + ["all"],
                        default="adaptive")
    parser.add_argument("--n",           type=int, nargs="+", default=[4, 8, 16, 32])
    parser.add_argument("--seeds",       type=int, default=5)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--config",      default="config/default.yaml")
    parser.add_argument("--device",      default="cpu")
    args = parser.parse_args()

    experiments = (
        list(EXPERIMENT_SCRIPTS.keys())
        if args.experiment == "all"
        else [args.experiment]
    )

    exit_code = 0
    for exp in experiments:
        script = EXPERIMENT_SCRIPTS[exp]
        print(f"\n{'='*60}")
        print(f" Running experiment: {exp}")
        print(f"{'='*60}")
        code = run_script(
            script=script,
            n_values=args.n,
            seeds=args.seeds,
            results_dir=args.results_dir,
            config=args.config,
            device=args.device,
        )
        if code != 0:
            print(f"[run_all_seeds] Experiment '{exp}' exited with code {code}")
            exit_code = code

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
