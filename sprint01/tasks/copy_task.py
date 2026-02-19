"""
Variable-Length Copy Task
--------------------------
Input: sequence of length L drawn from a vocabulary, followed by a separator symbol.
Output: reproduce the sequence exactly.

Difficulty ladder: L = 5, 10, 20, 50
Expected: slot count should correlate with L.
"""

import torch
import random
from typing import Tuple


class CopyTask:
    """
    Variable-length sequence copy task generator.

    Each episode:
      1. Sample a sequence of L symbols from {0, ..., vocab_size-1}.
      2. Append a separator token (vocab_size) to mark end of input.
      3. Target: reproduce the original L symbols.

    Input dim  = vocab_size + 1  (symbols + separator)
    Output dim = vocab_size      (predict each symbol)
    """

    def __init__(self, seq_len: int = 10, vocab_size: int = 8, device: str = "cpu"):
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.sep_token = vocab_size             # separator index
        self.input_dim = vocab_size + 1         # symbols + separator
        self.output_dim = vocab_size
        self.device = device

    def generate_batch(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            inputs:  [batch, seq_len+1, input_dim]  (sequence + separator)
            targets: [batch, seq_len, output_dim]   (original sequence, one-hot)
        """
        input_len = self.seq_len + 1  # sequence + separator
        inputs  = torch.zeros(batch_size, input_len, self.input_dim,  device=self.device)
        targets = torch.zeros(batch_size, self.seq_len, self.output_dim, device=self.device)

        for b in range(batch_size):
            seq = [random.randrange(self.vocab_size) for _ in range(self.seq_len)]

            # Encode sequence as one-hot
            for t, sym in enumerate(seq):
                inputs[b, t, sym] = 1.0
                targets[b, t, sym] = 1.0

            # Separator at position seq_len
            inputs[b, self.seq_len, self.sep_token] = 1.0

        return inputs, targets

    def compute_accuracy(self, logits: torch.Tensor, targets: torch.Tensor) -> float:
        """
        Args:
            logits:  [batch, seq_len, output_dim]
            targets: [batch, seq_len, output_dim] one-hot
        Returns:
            per-symbol accuracy (fraction of symbols correct across batch and positions)
        """
        pred = logits.argmax(dim=-1)      # [batch, seq_len]
        true = targets.argmax(dim=-1)     # [batch, seq_len]
        return (pred == true).float().mean().item()

    def with_len(self, seq_len: int) -> "CopyTask":
        """Return a new task with a different sequence length (difficulty)."""
        return CopyTask(seq_len=seq_len, vocab_size=self.vocab_size, device=self.device)
