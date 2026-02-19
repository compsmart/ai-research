"""
Curriculum Associative Recall
------------------------------
Fixes the core failure mode from Sprint 01: random per-episode key-value
assignments give the persistent memory bank nothing cross-episode to learn.

Design:
  K concepts are fixed for the entire experiment lifetime (same dict always).
  Each episode shows n_shown < K pairs then a query from the FULL K concepts.
  When the query is about an UNSEEN concept (probability (K-n_shown)/K),
  the LSTM alone cannot answer — it must retrieve from persistent memory.

Why this enables adaptive capacity:
  - After enough episodes, the model should cache all K concepts in memory.
  - Slot count at convergence should plateau near K - n_shown
    (the persistently-unseen fraction that must come from memory).
  - Higher K → more slots needed → direct testable hypothesis.
  - Lower n_shown → higher memory pressure → faster growth signal.

Difficulty ladder:
  K = 4, 8, 16, 32  (n_shown = K // 2 by default, so memory needs K//2 slots)

Variants:
  n_shown = 0  → "memory only" mode: all retrieval from persistent memory
  n_shown = K  → "LSTM only" mode: memory never needed (baseline ablation)
"""

import random
import torch
from typing import Optional, Tuple


class CurriculumRecall:
    """
    Fixed-concept partial-cue associative recall.

    Attributes
    ----------
    K          : int   — number of fixed concepts
    n_shown    : int   — pairs shown per episode (< K)
    vocab_size : int   — size of key/value vocabulary
    key_dim    : int   — one-hot key dimension (>= vocab_size)
    val_dim    : int   — one-hot value dimension (>= vocab_size)
    seed       : int   — controls the fixed concept dictionary

    The concept dictionary is created once in __init__ from `seed` and never
    changes. Training and evaluation must use the same instance (or same seed)
    for the dictionary to be consistent.
    """

    def __init__(
        self,
        K: int = 8,
        n_shown: int = 4,
        vocab_size: int = 32,
        key_dim: int = 32,
        val_dim: int = 32,
        concept_seed: int = 0,
        device: str = "cpu",
    ):
        assert K <= vocab_size,        f"K={K} must be <= vocab_size={vocab_size}"
        assert n_shown < K,           f"n_shown={n_shown} must be < K={K} (else no memory needed)"
        assert vocab_size <= key_dim, f"vocab_size={vocab_size} must be <= key_dim={key_dim}"
        assert vocab_size <= val_dim, f"vocab_size={vocab_size} must be <= val_dim={val_dim}"

        self.K = K
        self.n_shown = n_shown
        self.vocab_size = vocab_size
        self.key_dim = key_dim
        self.val_dim = val_dim
        self.input_dim = key_dim + val_dim   # [key_onehot | val_onehot]
        self.output_dim = val_dim
        self.device = device

        # Fixed concept dictionary — created once from concept_seed
        rng = random.Random(concept_seed)
        concept_keys = rng.sample(range(vocab_size), K)
        concept_vals = [rng.randrange(vocab_size) for _ in range(K)]
        self.concept_keys = concept_keys           # list[int], length K
        self.concept_vals = concept_vals           # list[int], length K
        # concept_dict[key_idx] = value_idx
        self.concept_dict = dict(zip(concept_keys, concept_vals))

        # Statistics for experiment logging
        self.memory_query_rate = (K - n_shown) / K  # fraction of queries that require memory

    # ------------------------------------------------------------------
    # Episode generation
    # ------------------------------------------------------------------

    def generate_batch(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        inputs  : [batch, n_shown + 1, input_dim]
            Sequence of n_shown (key, value) pairs then one query key.
            Query value portion is zeroed out.
        targets : [batch, val_dim]
            One-hot target value for the query key.

        Notes
        -----
        Each batch item independently samples:
          - Which n_shown concepts to show (random subset of K)
          - Which concept to query (from ALL K, including unseen)
        The query may therefore require persistent memory retrieval.
        """
        seq_len = self.n_shown + 1
        inputs  = torch.zeros(batch_size, seq_len, self.input_dim,  device=self.device)
        targets = torch.zeros(batch_size, self.val_dim,              device=self.device)

        for b in range(batch_size):
            # Randomly choose which n_shown concepts to present this episode
            shown_indices = random.sample(range(self.K), self.n_shown)
            shown_pairs   = [(self.concept_keys[i], self.concept_vals[i])
                             for i in shown_indices]

            # Fill in the shown (key, value) pairs
            for t, (k, v) in enumerate(shown_pairs):
                inputs[b, t, k] = 1.0                # key one-hot
                inputs[b, t, self.key_dim + v] = 1.0 # value one-hot

            # Query: randomly from ALL K concepts (may be unseen this episode)
            query_idx = random.randrange(self.K)
            query_key = self.concept_keys[query_idx]
            query_val = self.concept_vals[query_idx]
            inputs[b, self.n_shown, query_key] = 1.0  # query key only; value portion = 0
            targets[b, query_val] = 1.0

        return inputs, targets

    def generate_memory_only_batch(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        All-memory variant: show NO pairs, query from all K concepts.
        Tests pure memory retrieval without any LSTM in-episode context.
        seq_len = 1 (just the query).
        """
        inputs  = torch.zeros(batch_size, 1, self.input_dim,  device=self.device)
        targets = torch.zeros(batch_size, self.val_dim,         device=self.device)

        for b in range(batch_size):
            query_idx = random.randrange(self.K)
            query_key = self.concept_keys[query_idx]
            query_val = self.concept_vals[query_idx]
            inputs[b, 0, query_key] = 1.0
            targets[b, query_val] = 1.0

        return inputs, targets

    def compute_accuracy(self, logits: torch.Tensor,
                         targets: torch.Tensor) -> float:
        pred = logits.argmax(dim=-1)
        true = targets.argmax(dim=-1)
        return (pred == true).float().mean().item()

    # ------------------------------------------------------------------
    # Difficulty scaling
    # ------------------------------------------------------------------

    def with_K(self, K: int, n_shown: Optional[int] = None,
               concept_seed: int = 0) -> "CurriculumRecall":
        """Return a new task with different K (difficulty)."""
        n_shown = n_shown if n_shown is not None else K // 2
        return CurriculumRecall(
            K=K,
            n_shown=n_shown,
            vocab_size=max(self.vocab_size, K),
            key_dim=self.key_dim,
            val_dim=self.val_dim,
            concept_seed=concept_seed,
            device=self.device,
        )

    # ------------------------------------------------------------------
    # Concept dict access (for memory slot inspection)
    # ------------------------------------------------------------------

    def get_concept_tensors(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the full concept dictionary as one-hot tensors.

        Returns
        -------
        keys   : [K, key_dim]  one-hot key representations
        values : [K, val_dim]  one-hot value representations
        """
        keys   = torch.zeros(self.K, self.key_dim,  device=self.device)
        values = torch.zeros(self.K, self.val_dim,   device=self.device)
        for i, (k, v) in enumerate(zip(self.concept_keys, self.concept_vals)):
            keys[i, k]   = 1.0
            values[i, v] = 1.0
        return keys, values

    def __repr__(self) -> str:
        return (f"CurriculumRecall(K={self.K}, n_shown={self.n_shown}, "
                f"vocab={self.vocab_size}, memory_query_rate={self.memory_query_rate:.2f})")
