"""
Adaptive Model v3 — Step-by-step Write for Variable-Dict Recall
-----------------------------------------------------------------
Designed for VariableRecall: all K pairs are shown per episode,
concept dict is FRESH each batch (no weight memorization possible).

Key architectural change vs v2:
  1. Memory resets before each forward pass (episodic, not persistent)
  2. Support pairs are written to memory from sample 0 of each batch
     (all batch samples share the same concept dict)
  3. Queries for all batch samples are answered from shared memory

The CAPACITY ADVANTAGE over FixedLSTM:
  - FixedLSTM must compress K pairs into hidden_dim vectors
    (information bottleneck at large K)
  - AdaptiveModelV3 stores K pairs in K explicit slots
    (no compression — each slot encodes exactly one concept)

Prediction:
  K=4:   Both models succeed (small K, fits in any hidden_dim)
  K=8:   FixedLSTM-32 degrades, AdaptiveV3-32 succeeds
  K=16:  FixedLSTM-32 fails, AdaptiveV3-32 maintains accuracy
  K=32:  FixedLSTM-32 fails badly, AdaptiveV3-32 succeeds

This is the genuine capacity bottleneck demonstration.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.memory_bank_v2 import MemoryBankV2


class FixedLSTM(nn.Module):
    """LSTM-only baseline. Must compress K pairs into hidden_dim."""
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.encoder     = nn.LSTM(input_dim, hidden_dim, num_layers=1, batch_first=True)
        self.output_head = nn.Linear(hidden_dim, output_dim)
        self._last_read  = None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.encoder(inputs)
        return self.output_head(h_n[-1])


class AdaptiveModelV3(nn.Module):
    """
    LSTM + MemoryBankV2 with episodic write.

    Memory is RESET each forward pass. All K support pairs from
    sample 0 are written to K slots (no_grad). The LSTM still
    processes the full sequence for its contribution to output.
    The query reads from the freshly-written slots.

    Gradient flow:
      - Through LSTM (encoder)
      - Through key_proj (for concept-key addressing)
      - Through output_head (combining LSTM hidden + memory context)
      NOT through memory write (no_grad) — slots are teacher-forced.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        key_dim: int = 32,
        max_slots: int = 200,
        d_key: int = 32,
        d_val: int = 32,
        temp: float = 0.1,
        novelty_threshold: float = 0.5,
        loss_floor: float = 0.0,     # always write during support phase
        write_lr: float = 1.0,       # fully overwrite (support is authoritative)
        usage_ema_decay: float = 0.95,
        min_usage: float = 0.0,      # never prune within-episode slots
        min_age: int = 0,
        prune_every: int = 10**9,    # effectively never prune
        merge_threshold: float = 1.0,
        merge_every: int = 10**9,
    ):
        super().__init__()

        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.key_dim    = key_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.memory = MemoryBankV2(
            max_slots=max_slots,
            d_key=d_key,
            d_val=d_val,
            temp=temp,
            novelty_threshold=novelty_threshold,
            loss_floor=loss_floor,
            usage_ema_decay=usage_ema_decay,
            min_usage=min_usage,
            min_age=min_age,
            prune_every=prune_every,
            merge_threshold=merge_threshold,
            merge_every=merge_every,
            write_lr=write_lr,
            hidden_dim=hidden_dim,
            query_dim=key_dim,
        )

        self.output_head = nn.Linear(hidden_dim + d_val, output_dim)
        self._last_read  = None

    def _write_support_pairs(self, inputs: torch.Tensor) -> None:
        """
        Write all K support pairs from sample 0 into memory (no_grad).
        All samples share the same concept dict, so sample 0 suffices.
        Support pairs: timesteps where val portion is non-zero.
        """
        with torch.no_grad():
            seq_len  = inputs.size(1) - 1  # exclude final query step
            for t in range(seq_len):
                key_part = inputs[0:1, t, :self.key_dim]          # [1, key_dim]
                val_part = inputs[0:1, t, self.key_dim:]           # [1, val_dim]
                if val_part.sum().item() < 0.5:
                    continue  # padding or empty
                rd = self.memory.read(key_part)
                self.memory.write(
                    query=rd["query"],
                    target_signal=val_part,
                    current_loss=1.0,        # always write support pairs
                    attn=rd["attn"],
                    max_attn=0.0,
                    max_cos=0.0,             # force write (ignore novelty)
                )

    def _reset_episode_memory(self) -> None:
        with torch.no_grad():
            self.memory.active_mask[:] = False
            self.memory.usage_ema[:]   = 0.0
            self.memory.slot_age[:]    = 0
        self.memory.reset_event_counts()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        inputs: [batch, seq_len, input_dim]
          seq_len = K + 1; last timestep is the query.
        """
        # Reset and write episodic memory from support pairs
        self._reset_episode_memory()
        self._write_support_pairs(inputs)

        # LSTM over full sequence
        _, (h_n, _) = self.encoder(inputs)
        final_h = h_n[-1]                                    # [batch, hidden_dim]

        # Read from memory using concept query key
        concept_query = inputs[:, -1, :self.key_dim]         # [batch, key_dim]
        read_out = self.memory.read(concept_query)
        self._last_read = read_out

        ctx = read_out["ctx"]                                 # [batch, d_val]
        combined = torch.cat([final_h, ctx], dim=-1)
        return self.output_head(combined)
