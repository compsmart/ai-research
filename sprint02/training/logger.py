"""
Structured JSON Logger
-----------------------
Writes one JSON line per epoch to results/<experiment>/<seed>/log.jsonl.
Adding logging retroactively loses runs — build it first.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional


class Logger:
    """
    Append-only JSONL logger. Each call to log() writes one line.

    Schema (per epoch):
    {
      "epoch": int,
      "step": int,
      "task_n": int,
      "seed": int,
      "loss": float,
      "accuracy": float,
      "active_slots": int,
      "avg_slot_usage": float,
      "slot_entropy": float,
      "grad_norm": float,
      "write_events": int,
      "prune_events": int,
      "merge_events": int,
      "nan_detected": bool
    }
    """

    def __init__(self, experiment_name: str, seed: int,
                 results_dir: str = "results", extra_info: Optional[Dict] = None):
        self.experiment_name = experiment_name
        self.seed = seed

        log_dir = Path(results_dir) / experiment_name / str(seed)
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = log_dir / "log.jsonl"
        self.config_path = log_dir / "config.json"

        # Save extra run info
        if extra_info:
            with open(self.config_path, "w") as f:
                json.dump(extra_info, f, indent=2)

        # Open log file in append mode
        self._fh = open(self.log_path, "a")

    def log(self, record: Dict[str, Any]) -> None:
        """Write a single log record as a JSON line."""
        record.setdefault("seed", self.seed)
        self._fh.write(json.dumps(record) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    # ------------------------------------------------------------------
    # Convenience: load all records from a log file
    # ------------------------------------------------------------------

    @staticmethod
    def load(log_path: str):
        """Load all records from a JSONL log file."""
        records = []
        with open(log_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    @staticmethod
    def load_experiment(results_dir: str, experiment_name: str):
        """Load all records for all seeds in an experiment."""
        base = Path(results_dir) / experiment_name
        all_records = {}
        if not base.exists():
            return all_records
        for seed_dir in sorted(base.iterdir()):
            if seed_dir.is_dir():
                log_file = seed_dir / "log.jsonl"
                if log_file.exists():
                    all_records[seed_dir.name] = Logger.load(str(log_file))
        return all_records
