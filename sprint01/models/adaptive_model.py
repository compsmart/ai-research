"""
Adaptive Modular Memory Model
------------------------------
AdaptiveModel: LSTM encoder (within-episode memory) + MemoryBank (cross-episode
persistent memory). The LSTM accumulates k-v pairs step by step within an
episode; the MemoryBank provides additional cross-episode capacity whose size
is controlled by the adaptive trigger.

Why LSTM not MLP:
  Associative recall requires within-episode memory. An MLP processes each
  timestep independently and has no mechanism to store the k-v pairs seen
  earlier in the same episode. An LSTM accumulates them in its hidden state,
  making the task learnable. The MemoryBank then adds capacity on top.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base_net import BaseNet
from models.memory_bank import MemoryBank


class AdaptiveModel(nn.Module):
    """
    AMM model with LSTM encoder (default) or MLP encoder (legacy).

    Forward pass:
      1. LSTM processes full input sequence [batch, seq_len, input_dim].
         Its final hidden state encodes all k-v pairs + the query (within-episode).
      2. Query MemoryBank with final hidden state.
      3. Concatenate [final_h, ctx] and project to output.
      4. Write trigger is handled externally in the trainer.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        n_layers: int = 2,
        dropout: float = 0.0,
        encoder: str = "lstm",         # "lstm" or "mlp"
        # Memory bank kwargs
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
    ):
        super().__init__()
        self.encoder_type = encoder
        self.hidden_dim = hidden_dim

        if encoder == "lstm":
            self.encoder = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=1,
                batch_first=True,
            )
        else:
            self.encoder = BaseNet(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=hidden_dim,
                n_layers=n_layers,
                dropout=dropout,
            )

        self.memory = MemoryBank(
            max_slots=max_slots,
            d_key=d_key,
            d_val=d_val,
            temp=temp,
            error_threshold=error_threshold,
            usage_ema_decay=usage_ema_decay,
            min_usage=min_usage,
            min_age=min_age,
            prune_every=prune_every,
            merge_threshold=merge_threshold,
            merge_every=merge_every,
            write_lr=write_lr,
            hidden_dim=hidden_dim,
        )

        # Output projection: hidden + memory context -> output
        self.output_head = nn.Linear(hidden_dim + d_val, output_dim)
        self.output_dim = output_dim
        self._last_read = None

    def _encode(self, inputs: torch.Tensor) -> torch.Tensor:
        """Encode input sequence -> [batch, hidden_dim] (final step)."""
        if self.encoder_type == "lstm":
            _, (h_n, _) = self.encoder(inputs)  # h_n: [1, batch, hidden_dim]
            return h_n[-1]                        # [batch, hidden_dim]
        else:
            hidden = self.encoder(inputs)         # [batch, seq_len, hidden_dim]
            return hidden[:, -1, :]

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: [batch, seq_len, input_dim]
        Returns:
            logits: [batch, output_dim]
        """
        final_h = self._encode(inputs)             # [batch, hidden_dim]
        read_out = self.memory.read(final_h)
        self._last_read = read_out

        ctx = read_out["ctx"]                       # [batch, d_val]
        combined = torch.cat([final_h, ctx], dim=-1)
        return self.output_head(combined)

    def forward_sequence(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Sequence-to-sequence forward for copy task.

        Args:
            inputs: [batch, seq_len+1, input_dim]  (sequence + separator)
        Returns:
            logits: [batch, seq_len, output_dim]
        """
        batch, total_len, _ = inputs.shape
        seq_len = total_len - 1

        if self.encoder_type == "lstm":
            lstm_out, _ = self.encoder(inputs)    # [batch, total_len, hidden_dim]
            all_logits = []
            for t in range(seq_len):
                h_t = lstm_out[:, t, :]           # [batch, hidden_dim]
                read_out = self.memory.read(h_t)
                if t == seq_len - 1:
                    self._last_read = read_out
                ctx = read_out["ctx"]
                combined = torch.cat([h_t, ctx], dim=-1)
                all_logits.append(self.output_head(combined))
        else:
            hidden = self.encoder(inputs)         # [batch, total_len, hidden_dim]
            all_logits = []
            for t in range(seq_len):
                h_t = hidden[:, t, :]
                read_out = self.memory.read(h_t)
                if t == seq_len - 1:
                    self._last_read = read_out
                ctx = read_out["ctx"]
                combined = torch.cat([h_t, ctx], dim=-1)
                all_logits.append(self.output_head(combined))

        return torch.stack(all_logits, dim=1)     # [batch, seq_len, output_dim]


class NTMLite(nn.Module):
    """
    NTM-lite baseline: LSTM encoder + fixed-size memory (n_slots always active).
    Structurally identical to AMM but memory never grows or prunes.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        n_slots: int = 20,
        hidden_dim: int = 128,
        n_layers: int = 1,
        d_key: int = 32,
        d_val: int = 64,
        temp: float = 0.1,
        encoder: str = "lstm",
    ):
        super().__init__()
        self.encoder_type = encoder
        self.hidden_dim = hidden_dim

        if encoder == "lstm":
            self.encoder = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=1,
                batch_first=True,
            )
        else:
            self.encoder = BaseNet(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=hidden_dim,
                n_layers=n_layers,
            )

        self.memory = MemoryBank(
            max_slots=n_slots,
            d_key=d_key,
            d_val=d_val,
            temp=temp,
            hidden_dim=hidden_dim,
        )
        self.memory.growth_disabled = True

        with torch.no_grad():
            self.memory.active_mask[:] = True
            nn.init.normal_(self.memory.slots_key, std=0.1)
            nn.init.normal_(self.memory.slots_value, std=0.1)

        self.output_head = nn.Linear(hidden_dim + d_val, output_dim)
        self._last_read = None

    def _encode(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.encoder_type == "lstm":
            _, (h_n, _) = self.encoder(inputs)
            return h_n[-1]
        else:
            return self.encoder(inputs)[:, -1, :]

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        final_h = self._encode(inputs)
        read_out = self.memory.read(final_h)
        self._last_read = read_out
        ctx = read_out["ctx"]
        combined = torch.cat([final_h, ctx], dim=-1)
        return self.output_head(combined)


class FixedMLP(nn.Module):
    """
    Fixed MLP baseline — no memory. Flattens the input sequence and predicts directly.
    """

    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        output_dim: int,
        hidden_dim: int = 128,
        n_layers: int = 2,
    ):
        super().__init__()
        self.seq_len = seq_len
        flat_dim = input_dim * seq_len
        self.net = BaseNet(
            input_dim=flat_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            n_layers=n_layers,
        )
        self._last_read = None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = inputs.shape
        flat = inputs.reshape(batch, -1)
        return self.net(flat)


class FixedLSTM(nn.Module):
    """
    Fixed LSTM baseline — LSTM encoder only, no memory bank.
    Direct apples-to-apples comparison against AdaptiveModel.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.output_head = nn.Linear(hidden_dim, output_dim)
        self._last_read = None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.encoder(inputs)
        return self.output_head(h_n[-1])


class SmallTransformer(nn.Module):
    """
    Small transformer baseline: 2 heads, 2 layers, d_model=64.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        d_model: int = 64,
        n_heads: int = 2,
        n_layers: int = 2,
        d_ff: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_head = nn.Linear(d_model, output_dim)
        self._last_read = None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(inputs)
        x = self.transformer(x)
        x = x[:, -1, :]
        return self.output_head(x)
