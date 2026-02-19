"""
Adaptive Model v2
------------------
Uses MemoryBankV2 (novelty trigger). API matches sprint01 AdaptiveModel
so the same trainer and experiment scripts work unchanged.

Also exports the baselines (FixedLSTM, NTMLite) with the sprint02 signature.
"""

import torch
import torch.nn as nn
from models.memory_bank_v2 import MemoryBankV2


# ---------------------------------------------------------------------------
# Baselines — defined directly to avoid cross-sprint import complexity
# ---------------------------------------------------------------------------

class FixedLSTM(nn.Module):
    """LSTM-only baseline: no memory bank."""
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.encoder    = nn.LSTM(input_dim, hidden_dim, num_layers=1, batch_first=True)
        self.output_head = nn.Linear(hidden_dim, output_dim)
        self._last_read  = None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.encoder(inputs)
        return self.output_head(h_n[-1])


class NTMLite(nn.Module):
    """LSTM + fixed-size memory (growth disabled). Matches sprint01 NTMLite."""
    def __init__(self, input_dim: int, output_dim: int,
                 n_slots: int = 20, hidden_dim: int = 128,
                 d_key: int = 32, d_val: int = 32, temp: float = 0.1,
                 encoder: str = "lstm"):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers=1, batch_first=True)
        self.memory  = MemoryBankV2(
            max_slots=n_slots, d_key=d_key, d_val=d_val, temp=temp,
            hidden_dim=hidden_dim,
        )
        self.memory.growth_disabled = True
        with torch.no_grad():
            self.memory.active_mask[:] = True
            nn.init.normal_(self.memory.slots_key,   std=0.1)
            nn.init.normal_(self.memory.slots_value,  std=0.1)
        self.output_head = nn.Linear(hidden_dim + d_val, output_dim)
        self._last_read  = None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.encoder(inputs)
        final_h      = h_n[-1]
        read_out     = self.memory.read(final_h)
        self._last_read = read_out
        ctx          = read_out["ctx"]
        return self.output_head(torch.cat([final_h, ctx], dim=-1))


class AdaptiveModelV2(nn.Module):
    """
    LSTM encoder + MemoryBankV2 (novelty trigger).

    Memory addressing uses the concept query key (last timestep of input, first
    key_dim dims) rather than the LSTM hidden state. This gives natural separation
    between different concepts: one-hot keys project to distinct directions via
    key_proj, so raw cosine similarities between different concept queries are
    near 0, making novelty_threshold=0.5 a meaningful boundary.

    d_val is set equal to output_dim so stored values are directly interpretable.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        # Concept key dimension (first key_dim dims of each input timestep)
        key_dim: int = 32,
        # Memory bank
        max_slots: int = 100,
        d_key: int = 32,
        d_val: int = 32,          # should match output_dim for clean semantics
        temp: float = 0.1,
        novelty_threshold: float = 0.5,
        loss_floor: float = 0.3,
        usage_ema_decay: float = 0.95,
        min_usage: float = 0.01,
        min_age: int = 50,
        prune_every: int = 100,
        merge_threshold: float = 0.95,
        merge_every: int = 1000,
        write_lr: float = 0.3,
    ):
        super().__init__()

        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        self.key_dim = key_dim

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
            query_dim=key_dim,      # key_proj: key_dim -> d_key (concept-keyed)
        )

        self.output_head = nn.Linear(hidden_dim + d_val, output_dim)
        self.hidden_dim  = hidden_dim
        self.output_dim  = output_dim
        self._last_read  = None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: [batch, seq_len, input_dim]
                    Last timestep [:,  -1, :key_dim] is the one-hot concept query.
        Returns:
            logits: [batch, output_dim]
        """
        _, (h_n, _) = self.encoder(inputs)
        final_h = h_n[-1]                                    # [batch, hidden_dim]

        # Use the concept query key for memory addressing (not LSTM hidden state).
        # Different concepts have orthogonal one-hot keys → low pairwise cosine
        # similarity after key_proj → novelty trigger works correctly.
        concept_query = inputs[:, -1, :self.key_dim]         # [batch, key_dim]

        read_out = self.memory.read(concept_query)
        self._last_read = read_out

        ctx = read_out["ctx"]                                 # [batch, d_val]
        combined = torch.cat([final_h, ctx], dim=-1)
        return self.output_head(combined)
