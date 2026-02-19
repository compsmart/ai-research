"""
Variable-Dict Recall Task
--------------------------
In-context associative recall with fresh concept dictionaries.

Key difference from CurriculumRecall:
  - Each generate_batch() call draws ONE new random concept dict.
    All batch samples share the SAME dict.
  - Weight memorization is impossible: the dict changes every call.
  - n_shown = K (all K pairs are in every episode — pure retrieval test).
  - Different samples have different ORDERINGS and different QUERY concepts,
    but the same underlying K->V mappings.

Task:  Given K (key, value) pairs in random order, then a query key,
       output the correct value.  Model must retrieve from in-context store.

Capacity bottleneck:
  - FixedLSTM must compress K pairs (seq_len=K, input=64-dim each) into
    hidden_dim.  At K >> hidden_dim, information is lost.
  - AdaptiveModelV3 writes each pair to an explicit slot.
    K slots hold K pairs with zero compression loss.
"""

import random
import torch
from typing import Tuple


class VariableRecall:
    """
    Fresh concept dict per batch, all K pairs shown, query from K.
    All samples in a batch share the same concept dict.
    """

    def __init__(
        self,
        K: int = 8,
        vocab_size: int = 32,
        key_dim: int = 32,
        val_dim: int = 32,
        device: str = "cpu",
    ):
        assert K <= vocab_size, "K={} must fit in vocab_size={}".format(K, vocab_size)
        self.K          = K
        self.vocab_size = vocab_size
        self.key_dim    = key_dim
        self.val_dim    = val_dim
        self.device     = device

        self.input_dim  = key_dim + val_dim
        self.output_dim = val_dim
        self.seq_len    = K + 1          # K support pairs + 1 query

    def generate_batch(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        One shared concept dict per call; each sample has:
          - All K pairs in a different random order
          - A different random query concept

        Returns:
            inputs:  [batch_size, K+1, key_dim+val_dim]
            targets: [batch_size, val_dim] one-hot value for query concept
        """
        # Fresh concept dict for this batch
        keys    = random.sample(range(self.vocab_size), self.K)
        vals    = [random.randrange(self.vocab_size) for _ in range(self.K)]
        concept = list(zip(keys, vals))

        inputs  = torch.zeros(batch_size, self.seq_len, self.input_dim)
        targets = torch.zeros(batch_size, self.val_dim)

        for b in range(batch_size):
            order = list(range(self.K))
            random.shuffle(order)
            for t, idx in enumerate(order):
                k, v = concept[idx]
                inputs[b, t, k]                = 1.0
                inputs[b, t, self.key_dim + v] = 1.0

            q_idx = random.randrange(self.K)
            q_key, q_val = concept[q_idx]
            inputs[b, -1, q_key] = 1.0
            targets[b, q_val]    = 1.0

        return inputs.to(self.device), targets.to(self.device)
