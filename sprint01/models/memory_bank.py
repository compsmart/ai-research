"""
Memory Bank
-----------
Core adaptive memory module. Pre-allocates max_slots with an activation mask
to avoid mid-training optimizer state issues (adding nn.Parameters mid-run
breaks Adam state). Slots are plain tensors registered as buffers; they are
updated in-place via the write/prune/merge operations (not through autograd).

Operations:
  read   -- cosine-similarity attention over active slots
  write  -- error-triggered slot activation OR soft-update of best slot
  prune  -- deactivate low-usage old slots
  merge  -- combine highly similar slot pairs
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class MemoryBank(nn.Module):
    def __init__(
        self,
        max_slots: int = 100,
        d_key: int = 32,
        d_val: int = 64,
        temp: float = 0.1,
        error_threshold: float = 0.5,
        usage_ema_decay: float = 0.95,
        min_usage: float = 0.01,
        min_age: int = 50,
        prune_every: int = 100,
        merge_threshold: float = 0.95,
        merge_every: int = 1000,
        write_lr: float = 0.1,
        hidden_dim: int = 64,
    ):
        super().__init__()

        self.max_slots = max_slots
        self.d_key = d_key
        self.d_val = d_val
        self.temp = temp
        self.error_threshold = error_threshold
        self.usage_ema_decay = usage_ema_decay
        self.min_usage = min_usage
        self.min_age = min_age
        self.prune_every = prune_every
        self.merge_threshold = merge_threshold
        self.merge_every = merge_every
        self.write_lr = write_lr

        # Learned projection from hidden state to key dimension
        self.key_proj = nn.Linear(hidden_dim, d_key, bias=False)

        # Slot tensors — NOT nn.Parameters (avoids optimizer state issues)
        # Registered as buffers so they move with .to(device) and appear in state_dict
        self.register_buffer("slots_key",  torch.zeros(max_slots, d_key))
        self.register_buffer("slots_value", torch.zeros(max_slots, d_val))
        self.register_buffer("active_mask", torch.zeros(max_slots, dtype=torch.bool))
        self.register_buffer("usage_ema",   torch.zeros(max_slots))
        self.register_buffer("slot_age",    torch.zeros(max_slots, dtype=torch.long))
        self.register_buffer("grad_utility", torch.zeros(max_slots))

        # Running mean loss (for error threshold comparison)
        self.register_buffer("running_mean_loss", torch.tensor(1.0))

        # Step counters
        self.step = 0
        self.growth_disabled = False  # Set True for static-memory experiments

        # Event counters (reset each epoch by trainer)
        self.write_events = 0
        self.prune_events = 0
        self.merge_events = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def read(self, hidden: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute context vector from memory via cosine-similarity attention.

        Args:
            hidden: [batch, hidden_dim]
        Returns:
            dict with:
              ctx:   [batch, d_val]   weighted sum of active slot values
              attn:  [batch, n_active] attention weights over active slots
              query: [batch, d_key]   query vector (for write trigger)
        """
        query = self.key_proj(hidden)  # [batch, d_key]
        n_active = self.active_mask.sum().item()

        if n_active == 0:
            ctx = torch.zeros(hidden.size(0), self.d_val, device=hidden.device)
            attn = torch.zeros(hidden.size(0), 0, device=hidden.device)
            return {"ctx": ctx, "attn": attn, "query": query}

        active_idx = self.active_mask.nonzero(as_tuple=True)[0]  # [n_active]
        keys   = self.slots_key[active_idx]    # [n_active, d_key]
        values = self.slots_value[active_idx]  # [n_active, d_val]

        # Cosine similarity: [batch, n_active]
        q_norm = F.normalize(query, dim=-1)           # [batch, d_key]
        k_norm = F.normalize(keys,  dim=-1)           # [n_active, d_key]
        sims = torch.mm(q_norm, k_norm.t()) / self.temp  # [batch, n_active]

        attn = torch.softmax(sims, dim=-1)            # [batch, n_active]
        ctx  = torch.mm(attn, values)                 # [batch, d_val]

        # Update usage EMA (mean across batch)
        with torch.no_grad():
            mean_attn = attn.mean(dim=0)  # [n_active]
            self.usage_ema[active_idx] = (
                self.usage_ema_decay * self.usage_ema[active_idx]
                + (1 - self.usage_ema_decay) * mean_attn
            )

        return {"ctx": ctx, "attn": attn, "query": query}

    def write(self, query: torch.Tensor, target_signal: torch.Tensor,
              current_loss: float, attn: torch.Tensor) -> None:
        """
        Error-triggered slot growth or soft-update of best-matching slot.

        Args:
            query:         [batch, d_key]  (mean across batch used for new slot init)
            target_signal: [batch, d_val]  (mean across batch used for new slot init)
            current_loss:  scalar float
            attn:          [batch, n_active] last attention weights
        """
        if self.growth_disabled:
            self._soft_update_best(attn, target_signal)
            self._maybe_age_slots()
            return

        n_active = self.active_mask.sum().item()
        high_error = current_loss > self.error_threshold * self.running_mean_loss.item()

        with torch.no_grad():
            if high_error and n_active < self.max_slots:
                # Activate the next available slot
                inactive_idx = (~self.active_mask).nonzero(as_tuple=True)[0]
                if len(inactive_idx) > 0:
                    slot_i = inactive_idx[0].item()
                    mean_query = query.detach().mean(dim=0)   # [d_key]
                    mean_target = target_signal.detach().mean(dim=0)  # [d_val]
                    noise = torch.randn_like(mean_query) * 0.01
                    self.slots_key[slot_i]   = mean_query + noise
                    self.slots_value[slot_i] = mean_target
                    self.active_mask[slot_i] = True
                    self.usage_ema[slot_i]   = 0.0
                    self.slot_age[slot_i]    = 0
                    self.write_events += 1
            else:
                # Soft-update the best-matching slot
                self._soft_update_best(attn, target_signal)

        self._maybe_age_slots()

    def prune(self) -> None:
        """Deactivate low-usage old slots."""
        with torch.no_grad():
            active_idx = self.active_mask.nonzero(as_tuple=True)[0]
            for i in active_idx:
                i = i.item()
                if (self.usage_ema[i] < self.min_usage
                        and self.slot_age[i] > self.min_age):
                    self.active_mask[i] = False
                    self.usage_ema[i] = 0.0
                    self.slot_age[i] = 0
                    self.prune_events += 1

    def merge(self) -> None:
        """Combine pairs of slots with very similar keys."""
        with torch.no_grad():
            active_idx = self.active_mask.nonzero(as_tuple=True)[0].tolist()
            to_deactivate = set()
            for ii, i in enumerate(active_idx):
                if i in to_deactivate:
                    continue
                for j in active_idx[ii + 1:]:
                    if j in to_deactivate:
                        continue
                    k_i = F.normalize(self.slots_key[i].unsqueeze(0), dim=-1)
                    k_j = F.normalize(self.slots_key[j].unsqueeze(0), dim=-1)
                    sim = torch.mm(k_i, k_j.t()).item()
                    if sim > self.merge_threshold:
                        # Merge j into i
                        self.slots_key[i]   = (self.slots_key[i] + self.slots_key[j]) / 2
                        self.slots_value[i] = (self.slots_value[i] + self.slots_value[j]) / 2
                        self.usage_ema[i]   = max(self.usage_ema[i].item(),
                                                   self.usage_ema[j].item())
                        to_deactivate.add(j)
                        self.merge_events += 1

            for j in to_deactivate:
                self.active_mask[j] = False
                self.usage_ema[j] = 0.0
                self.slot_age[j] = 0

    def step_update(self, current_loss: float) -> None:
        """Call once per training step to update running loss and trigger prune/merge."""
        self.step += 1
        # Update running mean loss with EMA
        with torch.no_grad():
            alpha = 0.01  # slow adaptation
            self.running_mean_loss = (
                (1 - alpha) * self.running_mean_loss + alpha * current_loss
            )

        if self.step % self.prune_every == 0:
            self.prune()
        if self.step % self.merge_every == 0:
            self.merge()

    def reset_event_counts(self) -> None:
        """Call at the start of each epoch to reset per-epoch event counters."""
        self.write_events = 0
        self.prune_events = 0
        self.merge_events = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def active_count(self) -> int:
        return int(self.active_mask.sum().item())

    @property
    def avg_usage(self) -> float:
        n = self.active_count
        if n == 0:
            return 0.0
        return float(self.usage_ema[self.active_mask].mean().item())

    @property
    def slot_entropy(self) -> float:
        """Utilization entropy H = -sum(u_i * log(u_i)) over active slots."""
        n = self.active_count
        if n == 0:
            return 0.0
        u = self.usage_ema[self.active_mask]
        u = u / (u.sum() + 1e-8)
        entropy = -(u * (u + 1e-8).log()).sum().item()
        return entropy

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _soft_update_best(self, attn: torch.Tensor, target_signal: torch.Tensor) -> None:
        """Soft-update the slot with highest mean attention toward the target signal."""
        if attn.numel() == 0:
            return
        with torch.no_grad():
            best_local = attn.mean(dim=0).argmax().item()  # index in active slots
            active_idx = self.active_mask.nonzero(as_tuple=True)[0]
            if best_local >= len(active_idx):
                return
            best_global = active_idx[best_local].item()
            mean_target = target_signal.detach().mean(dim=0)
            self.slots_value[best_global] = (
                (1 - self.write_lr) * self.slots_value[best_global]
                + self.write_lr * mean_target
            )

    def _maybe_age_slots(self) -> None:
        """Increment age of all active slots."""
        with torch.no_grad():
            self.slot_age[self.active_mask] += 1
