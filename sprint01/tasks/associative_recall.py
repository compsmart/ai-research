"""
Associative Recall Task
-----------------------
Generate N (key, value) pairs followed by a query key. The model must recall
the value associated with the query key.

Difficulty ladder: N = 4, 8, 16, 32
Allows direct comparison to NTM paper results.
"""

import torch
import random
from typing import Tuple


class AssociativeRecall:
    """
    Associative recall task generator.

    Each episode:
      1. Sample N distinct key indices from vocab.
      2. Assign each key a random value index.
      3. Choose a query key (one of the N keys).
      4. Target: the value associated with the query key.

    Input representation: sequence of (key_onehot, value_onehot) pairs,
    then (query_onehot, zeros). Input dim = key_dim + val_dim.
    """

    def __init__(self, n: int = 8, vocab_size: int = 8,
                 key_dim: int = 8, val_dim: int = 8, device: str = "cpu"):
        # vocab_size must fit within both key and value one-hot dimensions
        vocab_size = min(vocab_size, key_dim, val_dim)
        assert n <= vocab_size, f"n={n} must be <= vocab_size={vocab_size}"
        self.n = n
        self.vocab_size = vocab_size
        self.key_dim = key_dim
        self.val_dim = val_dim
        self.input_dim = key_dim + val_dim
        self.output_dim = val_dim
        self.device = device

    def generate_batch(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            inputs:  [batch, seq_len, input_dim]  where seq_len = n + 1
            targets: [batch, val_dim]  one-hot target value
        """
        seq_len = self.n + 1  # n pairs + 1 query
        inputs = torch.zeros(batch_size, seq_len, self.input_dim, device=self.device)
        targets = torch.zeros(batch_size, self.val_dim, device=self.device)

        for b in range(batch_size):
            # Sample n distinct keys
            keys = random.sample(range(self.vocab_size), self.n)
            # Assign random values (may repeat)
            values = [random.randrange(self.vocab_size) for _ in range(self.n)]
            key_val = dict(zip(keys, values))

            # Fill in (key, value) pairs
            for i, (k, v) in enumerate(key_val.items()):
                inputs[b, i, k] = 1.0                    # key one-hot in first key_dim dims
                inputs[b, i, self.key_dim + v] = 1.0     # value one-hot in last val_dim dims

            # Query: pick a random key from the episode
            query_key = random.choice(keys)
            inputs[b, self.n, query_key] = 1.0            # query one-hot (no value portion)

            # Target: the value associated with the query key
            targets[b, key_val[query_key]] = 1.0

        return inputs, targets

    def compute_accuracy(self, logits: torch.Tensor, targets: torch.Tensor) -> float:
        """
        Args:
            logits:  [batch, val_dim]
            targets: [batch, val_dim] one-hot
        Returns:
            fraction correct
        """
        pred = logits.argmax(dim=-1)
        true = targets.argmax(dim=-1)
        return (pred == true).float().mean().item()

    def with_n(self, n: int) -> "AssociativeRecall":
        """Return a new task with a different N (difficulty)."""
        return AssociativeRecall(
            n=n,
            vocab_size=min(max(self.vocab_size, n), self.key_dim, self.val_dim),
            key_dim=self.key_dim,
            val_dim=self.val_dim,
            device=self.device,
        )
