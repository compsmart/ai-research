"""
Curriculum Recall — Unit Tests
--------------------------------
Run with:
    pytest sprint02/tests/test_curriculum.py -v
(from the ai-research root)

Or from sprint02/:
    pytest tests/test_curriculum.py -v
"""

import pytest
import torch
import sys, os
# sprint01 first so sprint02 overrides it at position 0
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'sprint01'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tasks.curriculum_recall import CurriculumRecall
from models.memory_bank_v2 import MemoryBankV2
import torch.nn.functional as F


# ------------------------------------------------------------------
# CurriculumRecall task tests
# ------------------------------------------------------------------

class TestCurriculumRecall:

    def test_concept_dict_is_fixed(self):
        """Same seed -> same concept dictionary every time."""
        t1 = CurriculumRecall(K=8, n_shown=4, concept_seed=42)
        t2 = CurriculumRecall(K=8, n_shown=4, concept_seed=42)
        assert t1.concept_keys == t2.concept_keys
        assert t1.concept_vals == t2.concept_vals

    def test_concept_dict_differs_by_seed(self):
        t1 = CurriculumRecall(K=8, n_shown=4, concept_seed=0)
        t2 = CurriculumRecall(K=8, n_shown=4, concept_seed=1)
        assert t1.concept_keys != t2.concept_keys

    def test_batch_shape(self):
        task = CurriculumRecall(K=8, n_shown=4, vocab_size=32, key_dim=32, val_dim=32)
        inputs, targets = task.generate_batch(16)
        assert inputs.shape  == (16, 5, 64),  f"Got {inputs.shape}"  # n_shown+1=5, key+val=64
        assert targets.shape == (16, 32),      f"Got {targets.shape}"

    def test_target_is_one_hot(self):
        task = CurriculumRecall(K=8, n_shown=4)
        _, targets = task.generate_batch(32)
        assert (targets.sum(dim=-1) == 1).all(), "Targets must be one-hot"

    def test_target_matches_concept_dict(self):
        """Every target value must correspond to the fixed concept dictionary."""
        task = CurriculumRecall(K=8, n_shown=4, vocab_size=32, key_dim=32, val_dim=32)
        concept_dict = task.concept_dict

        inputs, targets = task.generate_batch(64)
        query_positions = inputs[:, -1, :32]  # key portion of query step

        for b in range(64):
            # Find which concept was queried
            key_idx = query_positions[b].argmax().item()
            expected_val = concept_dict.get(key_idx)
            if expected_val is not None:
                actual_val = targets[b].argmax().item()
                assert actual_val == expected_val, (
                    f"Batch {b}: query key {key_idx} -> expected val {expected_val}, "
                    f"got {actual_val}"
                )

    def test_shown_pairs_are_from_concept_dict(self):
        """Every shown pair must correspond to a real concept."""
        task = CurriculumRecall(K=8, n_shown=4, vocab_size=32, key_dim=32, val_dim=32)
        inputs, _ = task.generate_batch(32)
        # For each shown pair (positions 0..n_shown-1)
        for b in range(32):
            for t in range(task.n_shown):
                key_oh  = inputs[b, t, :32]
                val_oh  = inputs[b, t, 32:]
                if key_oh.sum() == 0:
                    continue  # padding
                k = key_oh.argmax().item()
                v = val_oh.argmax().item()
                assert k in task.concept_dict, f"Key {k} not in concept dict"
                assert task.concept_dict[k] == v, (
                    f"Shown pair {k}->{v} doesn't match concept dict "
                    f"{k}->{task.concept_dict[k]}"
                )

    def test_n_shown_must_be_less_than_K(self):
        with pytest.raises(AssertionError):
            CurriculumRecall(K=4, n_shown=4)  # n_shown must be < K

    def test_K_must_fit_in_vocab(self):
        with pytest.raises(AssertionError):
            CurriculumRecall(K=64, n_shown=32, vocab_size=32)

    def test_memory_only_batch_shape(self):
        task = CurriculumRecall(K=8, n_shown=4)
        inputs, targets = task.generate_memory_only_batch(16)
        assert inputs.shape == (16, 1, 64)   # seq_len=1 (query only)
        assert targets.shape == (16, 32)

    def test_difficulty_ladder(self):
        base = CurriculumRecall(K=4, n_shown=2, vocab_size=32, key_dim=32, val_dim=32)
        for K in [8, 16, 32]:
            t = base.with_K(K)
            assert t.K == K
            assert t.n_shown == K // 2
            inp, tgt = t.generate_batch(8)
            assert inp.shape == (8, K // 2 + 1, 64)

    def test_memory_query_rate(self):
        """memory_query_rate = (K - n_shown) / K."""
        task = CurriculumRecall(K=8, n_shown=4)
        assert abs(task.memory_query_rate - 0.5) < 1e-6

        task2 = CurriculumRecall(K=8, n_shown=2)
        assert abs(task2.memory_query_rate - 0.75) < 1e-6

    def test_get_concept_tensors(self):
        task = CurriculumRecall(K=8, n_shown=4, vocab_size=32, key_dim=32, val_dim=32)
        keys, vals = task.get_concept_tensors()
        assert keys.shape  == (8, 32)
        assert vals.shape  == (8, 32)
        assert (keys.sum(dim=-1) == 1).all(), "Concept keys must be one-hot"
        assert (vals.sum(dim=-1) == 1).all(), "Concept vals must be one-hot"


# ------------------------------------------------------------------
# MemoryBankV2 novelty trigger tests
# ------------------------------------------------------------------

HIDDEN_DIM = 128
D_KEY = 32
D_VAL = 32
MAX_SLOTS = 20


def make_bank_v2(**kwargs):
    defaults = dict(
        max_slots=MAX_SLOTS, d_key=D_KEY, d_val=D_VAL,
        temp=0.1, novelty_threshold=0.5, loss_floor=0.3,
        usage_ema_decay=0.95, min_usage=0.01, min_age=50,
        prune_every=100, merge_threshold=0.95, merge_every=1000,
        write_lr=0.3, hidden_dim=HIDDEN_DIM,
    )
    defaults.update(kwargs)
    return MemoryBankV2(**defaults)


class TestMemoryBankV2:

    def test_novelty_trigger_fires_when_no_slots(self):
        """Should always grow when no slots exist (trivially novel)."""
        bank = make_bank_v2()
        assert bank.active_count == 0
        hidden = torch.randn(4, HIDDEN_DIM)
        out = bank.read(hidden)
        bank.write(out["query"], torch.randn(4, D_VAL),
                   current_loss=1.0, attn=out["attn"],
                   max_attn=out["max_attn"], max_cos=out.get("max_cos", 0.0))
        assert bank.active_count == 1
        assert bank.write_events == 1
        assert bank.novelty_fires == 1

    def test_novelty_trigger_suppressed_when_familiar(self):
        """With a well-matched slot, familiar_hits should increment instead."""
        bank = make_bank_v2(novelty_threshold=0.5)
        # Activate a slot and force its key to match the query perfectly
        bank.active_mask[0] = True
        hidden = torch.randn(1, HIDDEN_DIM)
        query = bank.key_proj(hidden)                       # [1, d_key]
        bank.slots_key[0] = F.normalize(query.detach()[0], dim=-1)
        bank.usage_ema[0] = 0.1
        bank.slot_age[0]  = 100

        # Read will now give high cosine similarity to slot 0
        out = bank.read(hidden)
        # max_cos (raw cosine sim) should be close to 1.0 with perfectly matching key
        assert out["max_cos"] > 0.5, f"Expected high max_cos, got {out['max_cos']:.4f}"

        initial_count = bank.active_count
        bank.write(out["query"], torch.randn(1, D_VAL),
                   current_loss=1.0, attn=out["attn"],
                   max_attn=out["max_attn"], max_cos=out["max_cos"])
        assert bank.active_count == initial_count, "Should not grow when familiar"
        assert bank.familiar_hits == 1

    def test_loss_floor_prevents_growth_when_already_correct(self):
        """No growth when loss is below loss_floor even if query is novel."""
        bank = make_bank_v2(loss_floor=0.5)
        hidden = torch.randn(4, HIDDEN_DIM)
        out = bank.read(hidden)
        bank.write(out["query"], torch.randn(4, D_VAL),
                   current_loss=0.1,         # below loss_floor=0.5
                   attn=out["attn"], max_attn=0.0, max_cos=0.0)
        assert bank.active_count == 0, "Should not grow when loss < loss_floor"

    def test_growth_stops_after_max_slots(self):
        """active_count must never exceed max_slots."""
        bank = make_bank_v2(max_slots=5, novelty_threshold=1.0)  # always novel (max_cos < 1.0)
        for _ in range(20):
            hidden = torch.randn(4, HIDDEN_DIM)
            out = bank.read(hidden)
            bank.write(out["query"], torch.randn(4, D_VAL),
                       current_loss=1.0, attn=out["attn"],
                       max_attn=out.get("max_attn", 0.0),
                       max_cos=out.get("max_cos", 0.0))
        assert bank.active_count <= 5

    def test_novelty_ratio_property(self):
        bank = make_bank_v2()
        # Force 3 novelty fires and 1 familiar hit manually
        bank.novelty_fires = 3
        bank.familiar_hits = 1
        assert abs(bank.novelty_ratio - 0.75) < 1e-6

    def test_read_returns_max_cos(self):
        bank = make_bank_v2()
        bank.active_mask[:3] = True
        bank.slots_key[:3] = torch.randn(3, D_KEY)
        hidden = torch.randn(4, HIDDEN_DIM)
        out = bank.read(hidden)
        assert "max_attn" in out
        assert "max_cos" in out
        assert 0.0 <= out["max_attn"] <= 1.0
        assert -1.0 <= out["max_cos"] <= 1.0

    def test_soft_update_moves_value_toward_target(self):
        bank = make_bank_v2(write_lr=0.5)
        bank.active_mask[0] = True
        bank.slots_key[0]   = F.normalize(torch.ones(D_KEY), dim=-1)
        bank.slots_value[0] = torch.zeros(D_VAL)
        bank.slot_age[0]    = 100

        hidden = torch.randn(1, HIDDEN_DIM)
        # Force query to match slot 0 exactly — max_cos will be ~1.0 (familiar)
        bank.slots_key[0] = F.normalize(bank.key_proj(hidden).detach()[0], dim=-1)
        out = bank.read(hidden)

        target = torch.ones(1, D_VAL)
        bank.write(out["query"], target,
                   current_loss=0.5, attn=out["attn"],
                   max_attn=out["max_attn"], max_cos=out["max_cos"])

        # Value should have moved toward all-ones (soft update, not new slot)
        v = bank.slots_value[0]
        assert v.sum().item() > 0, "Value did not move toward target"
