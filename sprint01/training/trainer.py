"""
Trainer
-------
Training loop with:
  - Structured JSON logging (every eval_every steps)
  - Write trigger (error-triggered slot growth)
  - In-loop stability monitoring
  - Configurable for baseline, static-memory, and adaptive modes
"""

import math
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from typing import Any, Dict, Optional

from training.logger import Logger
from training.metrics import (
    compute_grad_norm, has_nan_gradients, has_nan_in_output,
    compute_loss_trend, compute_accuracy_from_logits
)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        task: Any,
        cfg: Dict,
        experiment_name: str,
        seed: int,
        results_dir: str = "results",
        task_n: int = 8,
        use_sequence_forward: bool = False,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.task = task
        self.cfg = cfg
        self.device = device
        self.task_n = task_n
        self.use_sequence_forward = use_sequence_forward

        training_cfg = cfg.get("training", {})
        self.lr          = training_cfg.get("lr", 1e-3)
        self.weight_decay = training_cfg.get("weight_decay", 1e-5)
        self.batch_size  = training_cfg.get("batch_size", 32)
        self.max_steps   = training_cfg.get("max_steps", 5000)
        self.eval_every  = training_cfg.get("eval_every", 100)
        self.log_every   = training_cfg.get("log_every", 100)
        self.loss_window = training_cfg.get("loss_window", 100)
        self.grad_clip   = training_cfg.get("grad_clip", 5.0)

        self.optimizer = optim.Adam(
            model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        self.criterion = nn.CrossEntropyLoss()

        self.logger = Logger(
            experiment_name=experiment_name,
            seed=seed,
            results_dir=results_dir,
            extra_info={"cfg": cfg, "task_n": task_n, "seed": seed,
                        "experiment": experiment_name},
        )

        self.loss_history: deque = deque(maxlen=self.loss_window)
        self.accuracy_history = []
        self.slot_count_history = []
        self.step = 0
        self.epoch = 0
        self.nan_count = 0

        # Results summary populated after training
        self.results: Dict = {}

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self) -> Dict:
        """Run training for max_steps. Returns results summary dict."""
        self.model.train()
        running_loss = 0.0
        running_acc  = 0.0
        epoch_steps  = 0
        epoch_write  = 0
        epoch_prune  = 0
        epoch_merge  = 0

        # Reset memory event counters
        if hasattr(self.model, "memory"):
            self.model.memory.reset_event_counts()

        for self.step in range(1, self.max_steps + 1):
            loss_val, acc_val, nan_detected = self._train_step()

            running_loss += loss_val
            running_acc  += acc_val
            epoch_steps  += 1
            if nan_detected:
                self.nan_count += 1

            # Collect slot events
            if hasattr(self.model, "memory"):
                epoch_write += self.model.memory.write_events
                epoch_prune += self.model.memory.prune_events
                epoch_merge += self.model.memory.merge_events
                self.model.memory.reset_event_counts()

            # Log and evaluate
            if self.step % self.log_every == 0:
                self.epoch += 1
                avg_loss = running_loss / epoch_steps
                avg_acc  = running_acc  / epoch_steps

                active_slots = self._get_active_slots()
                avg_usage    = self._get_avg_usage()
                slot_entropy = self._get_slot_entropy()
                grad_norm    = compute_grad_norm(self.model)

                self.accuracy_history.append(avg_acc)
                self.slot_count_history.append(active_slots)

                record = {
                    "epoch":         self.epoch,
                    "step":          self.step,
                    "task_n":        self.task_n,
                    "loss":          round(avg_loss, 6),
                    "accuracy":      round(avg_acc, 6),
                    "active_slots":  active_slots,
                    "avg_slot_usage": round(avg_usage, 6),
                    "slot_entropy":  round(slot_entropy, 6),
                    "grad_norm":     round(grad_norm, 6),
                    "write_events":  epoch_write,
                    "prune_events":  epoch_prune,
                    "merge_events":  epoch_merge,
                    "nan_detected":  nan_detected,
                }
                self.logger.log(record)

                # Reset epoch accumulators
                running_loss = 0.0
                running_acc  = 0.0
                epoch_steps  = 0
                epoch_write  = 0
                epoch_prune  = 0
                epoch_merge  = 0

        self.logger.close()
        self._build_results()
        return self.results

    # ------------------------------------------------------------------
    # Single training step
    # ------------------------------------------------------------------

    def _train_step(self):
        self.optimizer.zero_grad()

        # Generate batch
        inputs, targets = self.task.generate_batch(self.batch_size)
        inputs  = inputs.to(self.device)
        targets = targets.to(self.device)

        # Forward
        if self.use_sequence_forward and hasattr(self.model, "forward_sequence"):
            logits = self.model.forward_sequence(inputs)  # [batch, seq_len, out_dim]
            batch, seq_len, out_dim = logits.shape
            loss = self.criterion(
                logits.reshape(batch * seq_len, out_dim),
                targets.reshape(batch * seq_len, targets.size(-1)).argmax(dim=-1),
            )
        else:
            logits = self.model(inputs)  # [batch, out_dim]
            loss = self.criterion(logits, targets.argmax(dim=-1))

        nan_detected = has_nan_in_output(logits) or has_nan_in_output(loss)

        if not nan_detected:
            loss.backward()

            # Gradient clipping
            if self.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.optimizer.step()

            loss_val = loss.item()
            self.loss_history.append(loss_val)

            # Memory write trigger
            if (hasattr(self.model, "memory")
                    and hasattr(self.model, "_last_read")
                    and self.model._last_read is not None):
                read_out = self.model._last_read
                # Target signal: the ground-truth one-hot targets expanded to d_val
                # Use the model's memory d_val; pad/truncate if needed
                d_val = self.model.memory.d_val
                if targets.size(-1) == d_val:
                    target_signal = targets.float()
                else:
                    # Pad or truncate to match d_val
                    pad = d_val - targets.size(-1)
                    if pad > 0:
                        target_signal = torch.cat(
                            [targets.float(),
                             torch.zeros(targets.size(0), pad, device=self.device)],
                            dim=-1,
                        )
                    else:
                        target_signal = targets.float()[:, :d_val]

                self.model.memory.write(
                    query=read_out["query"],
                    target_signal=target_signal,
                    current_loss=loss_val,
                    attn=read_out["attn"],
                )
                self.model.memory.step_update(loss_val)
        else:
            loss_val = float("nan")

        with torch.no_grad():
            acc_val = compute_accuracy_from_logits(logits, targets)

        return loss_val, acc_val, nan_detected

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_active_slots(self) -> int:
        if hasattr(self.model, "memory"):
            return self.model.memory.active_count
        return 0

    def _get_avg_usage(self) -> float:
        if hasattr(self.model, "memory"):
            return self.model.memory.avg_usage
        return 0.0

    def _get_slot_entropy(self) -> float:
        if hasattr(self.model, "memory"):
            return self.model.memory.slot_entropy
        return 0.0

    def _build_results(self) -> None:
        """Populate self.results summary after training."""
        n_evals = len(self.accuracy_history)
        if n_evals == 0:
            self.results = {}
            return

        late_start = int(0.8 * n_evals)
        late_slots = self.slot_count_history[late_start:] if self.slot_count_history else [0]
        late_accs  = self.accuracy_history[late_start:] if self.accuracy_history else [0]

        self.results = {
            "final_loss":      float(list(self.loss_history)[-1]) if self.loss_history else float("nan"),
            "final_acc":       self.accuracy_history[-1] if self.accuracy_history else 0.0,
            "nan_count":       self.nan_count,
            "grad_norm_max":   max((r for r in [compute_grad_norm(self.model)]), default=0.0),
            "slot_count_max":  max(self.slot_count_history) if self.slot_count_history else 0,
            "slot_counts":     self.slot_count_history,
            "accuracy_history": self.accuracy_history,
            "late_slot_change": (
                (max(late_slots) - min(late_slots)) / max(late_slots)
                if late_slots and max(late_slots) > 0 else 0.0
            ),
            "task_n":          self.task_n,
        }
