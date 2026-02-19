"""
Trainer v2 — Curriculum Recall
--------------------------------
Extends sprint01 Trainer with:
  - Passes max_attn to the write trigger (required by MemoryBankV2)
  - Logs novelty_fires, familiar_hits, novelty_ratio per epoch
  - Correct target_signal dimension (d_val = output_dim, no padding needed)
  - Absolute loss floor trigger (no running mean required)

Imports sprint01 Logger and metrics unchanged.
"""

import math
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from typing import Any, Dict

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'sprint01'))

from training.logger import Logger
from training.metrics import compute_grad_norm, has_nan_in_output, compute_accuracy_from_logits


class TrainerV2:
    def __init__(
        self,
        model: nn.Module,
        task: Any,
        cfg: Dict,
        experiment_name: str,
        seed: int,
        results_dir: str = "results",
        task_K: int = 8,
        device: str = "cpu",
    ):
        self.model  = model.to(device)
        self.task   = task
        self.cfg    = cfg
        self.device = device
        self.task_K = task_K

        tr = cfg.get("training", {})
        self.lr           = tr.get("lr",           5e-4)
        self.weight_decay = tr.get("weight_decay", 1e-5)
        self.batch_size   = tr.get("batch_size",   32)
        self.max_steps    = tr.get("max_steps",    10000)
        self.log_every    = tr.get("log_every",    200)
        self.loss_window  = tr.get("loss_window",  200)
        self.grad_clip    = tr.get("grad_clip",    5.0)

        self.optimizer = optim.Adam(
            model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        self.criterion = nn.CrossEntropyLoss()

        self.logger = Logger(
            experiment_name=experiment_name,
            seed=seed,
            results_dir=results_dir,
            extra_info={"cfg": cfg, "task_K": task_K, "seed": seed},
        )

        self.loss_history:    deque = deque(maxlen=self.loss_window)
        self.accuracy_history = []
        self.slot_count_history = []
        self.step      = 0
        self.epoch     = 0
        self.nan_count = 0
        self.results:  Dict = {}

    # ------------------------------------------------------------------

    def train(self) -> Dict:
        self.model.train()
        running_loss  = 0.0
        running_acc   = 0.0
        epoch_steps   = 0
        epoch_write   = 0
        epoch_prune   = 0
        epoch_merge   = 0
        epoch_novelty = 0
        epoch_familiar = 0

        if hasattr(self.model, "memory"):
            self.model.memory.reset_event_counts()

        for self.step in range(1, self.max_steps + 1):
            loss_val, acc_val, nan_detected = self._train_step()

            running_loss += loss_val if not math.isnan(loss_val) else 0.0
            running_acc  += acc_val
            epoch_steps  += 1
            if nan_detected:
                self.nan_count += 1

            if hasattr(self.model, "memory"):
                m = self.model.memory
                epoch_write    += m.write_events
                epoch_prune    += m.prune_events
                epoch_merge    += m.merge_events
                epoch_novelty  += getattr(m, "novelty_fires", 0)
                epoch_familiar += getattr(m, "familiar_hits", 0)
                m.reset_event_counts()

            if self.step % self.log_every == 0:
                self.epoch  += 1
                avg_loss     = running_loss / max(epoch_steps, 1)
                avg_acc      = running_acc  / max(epoch_steps, 1)

                active_slots = self._get_active_slots()
                avg_usage    = self._get_avg_usage()
                slot_entropy = self._get_slot_entropy()
                novelty_ratio = (epoch_novelty / max(epoch_novelty + epoch_familiar, 1))
                grad_norm    = compute_grad_norm(self.model)

                self.accuracy_history.append(avg_acc)
                self.slot_count_history.append(active_slots)

                record = {
                    "epoch":          self.epoch,
                    "step":           self.step,
                    "task_K":         self.task_K,
                    "loss":           round(avg_loss, 6),
                    "accuracy":       round(avg_acc, 6),
                    "active_slots":   active_slots,
                    "avg_slot_usage": round(avg_usage, 6),
                    "slot_entropy":   round(slot_entropy, 6),
                    "grad_norm":      round(grad_norm, 6),
                    "write_events":   epoch_write,
                    "prune_events":   epoch_prune,
                    "merge_events":   epoch_merge,
                    "novelty_fires":  epoch_novelty,
                    "familiar_hits":  epoch_familiar,
                    "novelty_ratio":  round(novelty_ratio, 4),
                    "nan_detected":   nan_detected,
                }
                self.logger.log(record)

                running_loss   = 0.0
                running_acc    = 0.0
                epoch_steps    = 0
                epoch_write    = 0
                epoch_prune    = 0
                epoch_merge    = 0
                epoch_novelty  = 0
                epoch_familiar = 0

        self.logger.close()
        self._build_results()
        return self.results

    # ------------------------------------------------------------------

    def _train_step(self):
        self.optimizer.zero_grad()

        inputs, targets = self.task.generate_batch(self.batch_size)
        inputs  = inputs.to(self.device)
        targets = targets.to(self.device)

        logits = self.model(inputs)                          # [batch, output_dim]
        loss   = self.criterion(logits, targets.argmax(dim=-1))
        nan_detected = has_nan_in_output(logits) or has_nan_in_output(loss)

        if not nan_detected:
            loss.backward()
            if self.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

            loss_val = loss.item()
            self.loss_history.append(loss_val)

            # Write trigger — pass max_attn if available (MemoryBankV2)
            if (hasattr(self.model, "memory")
                    and hasattr(self.model, "_last_read")
                    and self.model._last_read is not None):
                read_out = self.model._last_read
                d_val    = self.model.memory.d_val

                # Target signal: match d_val (output_dim == d_val in sprint02)
                if targets.size(-1) == d_val:
                    target_signal = targets.float()
                elif targets.size(-1) < d_val:
                    pad = d_val - targets.size(-1)
                    target_signal = torch.cat(
                        [targets.float(),
                         torch.zeros(targets.size(0), pad, device=self.device)], dim=-1)
                else:
                    target_signal = targets.float()[:, :d_val]

                self.model.memory.write(
                    query=read_out["query"],
                    target_signal=target_signal,
                    current_loss=loss_val,
                    attn=read_out["attn"],
                    max_attn=read_out.get("max_attn", 0.0),
                    max_cos=read_out.get("max_cos", 0.0),
                )
                self.model.memory.step_update(loss_val)
        else:
            loss_val = float("nan")

        with torch.no_grad():
            acc_val = compute_accuracy_from_logits(logits, targets)

        return loss_val, acc_val, nan_detected

    # ------------------------------------------------------------------

    def _get_active_slots(self) -> int:
        return self.model.memory.active_count if hasattr(self.model, "memory") else 0

    def _get_avg_usage(self) -> float:
        return self.model.memory.avg_usage if hasattr(self.model, "memory") else 0.0

    def _get_slot_entropy(self) -> float:
        return self.model.memory.slot_entropy if hasattr(self.model, "memory") else 0.0

    def _build_results(self) -> None:
        n = len(self.accuracy_history)
        if n == 0:
            self.results = {}
            return
        late_start = int(0.8 * n)
        late_slots = self.slot_count_history[late_start:] or [0]
        self.results = {
            "final_loss":       float(list(self.loss_history)[-1]) if self.loss_history else float("nan"),
            "final_acc":        self.accuracy_history[-1] if self.accuracy_history else 0.0,
            "nan_count":        self.nan_count,
            "grad_norm_max":    compute_grad_norm(self.model),
            "slot_count_max":   max(self.slot_count_history) if self.slot_count_history else 0,
            "slot_counts":      self.slot_count_history,
            "accuracy_history": self.accuracy_history,
            "late_slot_change": (
                (max(late_slots) - min(late_slots)) / max(late_slots)
                if late_slots and max(late_slots) > 0 else 0.0
            ),
            "task_K":           self.task_K,
        }
