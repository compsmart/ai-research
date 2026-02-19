"""
Memory Bank v2 — Novelty-Based Write Trigger
----------------------------------------------
Sprint 01 diagnosis: the loss-relative trigger fires ~50% of every step
indefinitely (any step above running_mean * 0.5 triggers growth), creating
a perpetual write/prune cycle with no equilibrium.

Sprint 02 fix: replace with a NOVELTY trigger.
  - Grow when: max attention over active slots < novelty_threshold
    (no existing slot matches the current query well)
  - Do NOT grow when: some slot already matches well (familiar query)

Why this works:
  - After learning K concepts, K slots have high cosine similarity to the K
    concept keys. All future queries match one of them → max_attn stays high
    → growth stops naturally.
  - The trigger is structural (architecture-based), not loss-level-based.
    It fires on genuine novelty, not residual optimization noise.
  - With softmax temperature=0.1, a well-matched slot gets attn close to 1.0.
    A threshold of 0.5 clearly separates "matched" from "unmatched".

Secondary fix: absolute loss floor as a fallback when no slots exist yet.
  Both conditions must hold to grow: novelty AND loss above floor.

API is identical to memory_bank.py so existing trainers/models need no changes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class MemoryBankV2(nn.Module):
    def __init__(
        self,
        max_slots: int = 100,
        d_key: int = 32,
        d_val: int = 32,
        temp: float = 0.1,
        # Novelty trigger (primary)
        novelty_threshold: float = 0.5,  # grow if max_attn < this
        # Absolute loss floor (secondary — also required for growth)
        loss_floor: float = 0.3,         # grow only if loss > this
        # Usage EMA and pruning
        usage_ema_decay: float = 0.95,
        min_usage: float = 0.01,
        min_age: int = 50,
        prune_every: int = 100,
        # Slot merging
        merge_threshold: float = 0.95,
        merge_every: int = 1000,
        # Soft-update
        write_lr: float = 0.3,
        # Encoder output dimension (used only when query_dim is None)
        hidden_dim: int = 128,
        # Explicit query dimension for key_proj input.
        # When set, key_proj maps query_dim -> d_key (e.g. concept key dim).
        # When None, falls back to hidden_dim (original behaviour).
        query_dim: int = None,
    ):
        super().__init__()

        self.max_slots       = max_slots
        self.d_key           = d_key
        self.d_val           = d_val
        self.temp            = temp
        self.novelty_threshold = novelty_threshold
        self.loss_floor      = loss_floor
        self.usage_ema_decay = usage_ema_decay
        self.min_usage       = min_usage
        self.min_age         = min_age
        self.prune_every     = prune_every
        self.merge_threshold = merge_threshold
        self.merge_every     = merge_every
        self.write_lr        = write_lr

        proj_in = query_dim if query_dim is not None else hidden_dim
        self.key_proj = nn.Linear(proj_in, d_key, bias=False)

        self.register_buffer("slots_key",    torch.zeros(max_slots, d_key))
        self.register_buffer("slots_value",  torch.zeros(max_slots, d_val))
        self.register_buffer("active_mask",  torch.zeros(max_slots, dtype=torch.bool))
        self.register_buffer("usage_ema",    torch.zeros(max_slots))
        self.register_buffer("slot_age",     torch.zeros(max_slots, dtype=torch.long))

        self.step             = 0
        self.growth_disabled  = False

        self.write_events = 0
        self.prune_events = 0
        self.merge_events = 0
        # Track novelty trigger fires separately from growth events
        self.novelty_fires  = 0
        self.familiar_hits  = 0

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def read(self, hidden: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Cosine-similarity attention over active slots."""
        query = self.key_proj(hidden)            # [batch, d_key]
        n_active = self.active_mask.sum().item()

        if n_active == 0:
            ctx  = torch.zeros(hidden.size(0), self.d_val, device=hidden.device)
            attn = torch.zeros(hidden.size(0), 0,          device=hidden.device)
            return {"ctx": ctx, "attn": attn, "query": query, "max_attn": 0.0}

        active_idx = self.active_mask.nonzero(as_tuple=True)[0]
        keys   = self.slots_key[active_idx]
        values = self.slots_value[active_idx]

        q_norm = F.normalize(query, dim=-1)
        k_norm = F.normalize(keys,  dim=-1)
        cos_sims = torch.mm(q_norm, k_norm.t())           # [batch, n_active] raw cosine sims
        sims     = cos_sims / self.temp
        attn     = torch.softmax(sims, dim=-1)
        ctx      = torch.mm(attn, values)

        with torch.no_grad():
            mean_attn = attn.mean(dim=0)
            self.usage_ema[active_idx] = (
                self.usage_ema_decay * self.usage_ema[active_idx]
                + (1 - self.usage_ema_decay) * mean_attn
            )

        # max_attn: post-softmax (for logging/usage tracking)
        # max_cos:  raw cosine similarity of best-matching slot (for novelty trigger)
        max_attn = float(attn.max(dim=-1).values.mean().item())
        max_cos  = float(cos_sims.max(dim=-1).values.mean().item())
        return {"ctx": ctx, "attn": attn, "query": query,
                "max_attn": max_attn, "max_cos": max_cos}

    # ------------------------------------------------------------------
    # Write  (novelty trigger)
    # ------------------------------------------------------------------

    def write(self, query: torch.Tensor, target_signal: torch.Tensor,
              current_loss: float, attn: torch.Tensor,
              max_attn: float = 0.0, max_cos: float = 0.0) -> None:
        """
        Novelty-triggered slot growth.

        Grow a new slot when:
          (a) max RAW cosine similarity < novelty_threshold
              (no existing slot's key direction matches the current query well)
          (b) current loss exceeds the absolute floor   → model isn't already correct
          (c) active_count < max_slots                  → capacity available
          (d) growth is not disabled

        max_cos (raw cosine sim) is used, NOT post-softmax attention.
        With softmax, a single slot always gets attn=1.0 — post-softmax is
        useless for novelty detection. Raw cosine similarity is in [-1, 1]
        and meaningfully measures how well a stored key matches the current query.
        """
        if self.growth_disabled:
            self._soft_update_best(attn, target_signal)
            self._maybe_age_slots()
            return

        n_active    = self.active_count
        is_novel    = (n_active == 0) or (max_cos < self.novelty_threshold)
        above_floor = (current_loss > self.loss_floor)
        can_grow    = (n_active < self.max_slots)

        with torch.no_grad():
            if is_novel and above_floor and can_grow:
                self.novelty_fires += 1
                inactive_idx = (~self.active_mask).nonzero(as_tuple=True)[0]
                if len(inactive_idx) > 0:
                    slot_i = inactive_idx[0].item()
                    mean_q = query.detach().mean(dim=0)
                    mean_t = target_signal.detach().mean(dim=0)
                    noise  = torch.randn_like(mean_q) * 0.001
                    self.slots_key[slot_i]   = F.normalize(mean_q + noise, dim=-1)
                    self.slots_value[slot_i] = mean_t
                    self.active_mask[slot_i] = True
                    self.usage_ema[slot_i]   = 0.0
                    self.slot_age[slot_i]    = 0
                    self.write_events       += 1
            else:
                if not is_novel:
                    self.familiar_hits += 1
                self._soft_update_best(attn, target_signal)

        self._maybe_age_slots()

    # ------------------------------------------------------------------
    # Prune
    # ------------------------------------------------------------------

    def prune(self) -> None:
        with torch.no_grad():
            active_idx = self.active_mask.nonzero(as_tuple=True)[0]
            for i in active_idx:
                i = i.item()
                if (self.usage_ema[i] < self.min_usage
                        and self.slot_age[i] > self.min_age):
                    self.active_mask[i] = False
                    self.usage_ema[i]   = 0.0
                    self.slot_age[i]    = 0
                    self.prune_events  += 1

    # ------------------------------------------------------------------
    # Merge
    # ------------------------------------------------------------------

    def merge(self) -> None:
        with torch.no_grad():
            active_idx  = self.active_mask.nonzero(as_tuple=True)[0].tolist()
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
                        self.slots_key[i]   = (self.slots_key[i] + self.slots_key[j]) / 2
                        self.slots_value[i] = (self.slots_value[i] + self.slots_value[j]) / 2
                        self.usage_ema[i]   = max(self.usage_ema[i].item(),
                                                   self.usage_ema[j].item())
                        to_deactivate.add(j)
                        self.merge_events  += 1

            for j in to_deactivate:
                self.active_mask[j] = False
                self.usage_ema[j]   = 0.0
                self.slot_age[j]    = 0

    # ------------------------------------------------------------------
    # Step update
    # ------------------------------------------------------------------

    def step_update(self, current_loss: float) -> None:
        self.step += 1
        if self.step % self.prune_every == 0:
            self.prune()
        if self.step % self.merge_every == 0:
            self.merge()

    def reset_event_counts(self) -> None:
        self.write_events  = 0
        self.prune_events  = 0
        self.merge_events  = 0
        self.novelty_fires = 0
        self.familiar_hits = 0

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
        n = self.active_count
        if n == 0:
            return 0.0
        u = self.usage_ema[self.active_mask]
        u = u / (u.sum() + 1e-8)
        return float(-(u * (u + 1e-8).log()).sum().item())

    @property
    def novelty_ratio(self) -> float:
        """Fraction of write calls that fired the novelty trigger vs familiar."""
        total = self.novelty_fires + self.familiar_hits
        return self.novelty_fires / total if total > 0 else 0.0

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _soft_update_best(self, attn: torch.Tensor,
                          target_signal: torch.Tensor) -> None:
        if attn.numel() == 0:
            return
        with torch.no_grad():
            best_local  = attn.mean(dim=0).argmax().item()
            active_idx  = self.active_mask.nonzero(as_tuple=True)[0]
            if best_local >= len(active_idx):
                return
            best_global = active_idx[best_local].item()
            mean_target = target_signal.detach().mean(dim=0)
            self.slots_value[best_global] = (
                (1 - self.write_lr) * self.slots_value[best_global]
                + self.write_lr * mean_target
            )

    def _maybe_age_slots(self) -> None:
        with torch.no_grad():
            self.slot_age[self.active_mask] += 1
