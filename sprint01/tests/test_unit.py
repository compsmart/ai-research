"""
Unit Tests — Memory Bank Correctness
--------------------------------------
ALL tests must pass before any training code is run (Gate 1).

Run with:
    pytest tests/test_unit.py -v
"""

import pytest
import torch
import torch.nn.functional as F
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.memory_bank import MemoryBank


HIDDEN_DIM = 64
D_KEY = 32
D_VAL = 64
MAX_SLOTS = 20


def make_bank(**kwargs):
    defaults = dict(
        max_slots=MAX_SLOTS, d_key=D_KEY, d_val=D_VAL,
        temp=0.1, error_threshold=0.5, usage_ema_decay=0.95,
        min_usage=0.01, min_age=50, prune_every=100,
        merge_threshold=0.95, merge_every=1000,
        write_lr=0.5, hidden_dim=HIDDEN_DIM,
    )
    defaults.update(kwargs)
    return MemoryBank(**defaults)


def make_hidden(batch=4):
    return torch.randn(batch, HIDDEN_DIM)


# ------------------------------------------------------------------
# test_read_returns_weighted_sum
# ------------------------------------------------------------------

def test_read_returns_weighted_sum():
    """ctx must equal attn @ values over active slots."""
    bank = make_bank()
    batch = 4

    # Manually activate 3 slots with known values
    bank.active_mask[:3] = True
    bank.slots_key[:3] = F.normalize(torch.randn(3, D_KEY), dim=-1)
    bank.slots_value[:3] = torch.ones(3, D_VAL)  # all ones for easy verification

    hidden = make_hidden(batch)
    out = bank.read(hidden)

    ctx  = out["ctx"]   # [batch, D_VAL]
    attn = out["attn"]  # [batch, 3]

    # Reconstruct ctx manually
    values = bank.slots_value[:3]  # [3, D_VAL]
    expected = torch.mm(attn, values)

    assert ctx.shape == (batch, D_VAL), f"ctx shape wrong: {ctx.shape}"
    assert torch.allclose(ctx, expected, atol=1e-5), \
        f"ctx != attn @ values. Max diff: {(ctx - expected).abs().max()}"


# ------------------------------------------------------------------
# test_write_activates_new_slot_on_high_error
# ------------------------------------------------------------------

def test_write_activates_new_slot_on_high_error():
    """When loss is high (above threshold * running_mean), a new slot is activated."""
    bank = make_bank()
    bank.running_mean_loss = torch.tensor(1.0)

    initial_count = bank.active_count
    assert initial_count == 0

    hidden = make_hidden()
    out = bank.read(hidden)

    query  = out["query"]
    target = torch.randn(4, D_VAL)
    attn   = out["attn"]

    # High error: current_loss >> threshold * running_mean
    high_loss = 5.0  # well above 0.5 * 1.0
    bank.write(query, target, current_loss=high_loss, attn=attn)

    assert bank.active_count == 1, \
        f"Expected 1 active slot after high-error write, got {bank.active_count}"
    assert bank.write_events == 1


# ------------------------------------------------------------------
# test_write_updates_existing_slot_on_low_error
# ------------------------------------------------------------------

def test_write_updates_existing_slot_on_low_error():
    """When loss is low, an existing slot is soft-updated, not a new one created."""
    bank = make_bank(write_lr=0.5)
    bank.running_mean_loss = torch.tensor(1.0)

    # Pre-activate one slot
    bank.active_mask[0] = True
    original_value = torch.zeros(D_VAL)
    bank.slots_value[0] = original_value.clone()
    bank.slots_key[0]   = F.normalize(torch.ones(D_KEY), dim=-1)
    bank.slot_age[0]    = 100

    hidden = make_hidden(1)
    out = bank.read(hidden)
    query  = out["query"]
    target = torch.ones(1, D_VAL)  # known target
    attn   = out["attn"]

    # Low error: well below threshold
    low_loss = 0.01
    bank.write(query, target, current_loss=low_loss, attn=attn)

    # Slot count must not increase
    assert bank.active_count == 1, \
        f"Expected 1 active slot (no growth on low error), got {bank.active_count}"

    # Value must have moved toward target
    updated = bank.slots_value[0]
    diff_from_zero   = updated.abs().sum().item()
    diff_from_target = (updated - torch.ones(D_VAL)).abs().sum().item()
    assert diff_from_zero > 0, "Value unchanged after soft-update"
    assert diff_from_target < D_VAL, "Value did not move toward target"


# ------------------------------------------------------------------
# test_prune_removes_low_usage_old_slots
# ------------------------------------------------------------------

def test_prune_removes_low_usage_old_slots():
    """Low-usage slots older than min_age must be deactivated by prune()."""
    bank = make_bank(min_usage=0.05, min_age=10)

    # Activate 3 slots: 2 with low usage, 1 with high usage
    bank.active_mask[:3] = True
    bank.usage_ema[0] = 0.001   # low — should be pruned
    bank.usage_ema[1] = 0.001   # low — should be pruned
    bank.usage_ema[2] = 0.5     # high — keep
    bank.slot_age[:3] = 100     # old enough

    bank.prune()

    assert not bank.active_mask[0], "Slot 0 should have been pruned"
    assert not bank.active_mask[1], "Slot 1 should have been pruned"
    assert bank.active_mask[2],     "Slot 2 should be kept (high usage)"
    assert bank.active_count == 1


# ------------------------------------------------------------------
# test_prune_preserves_young_slots
# ------------------------------------------------------------------

def test_prune_preserves_young_slots():
    """Slots younger than min_age must be protected even if usage is low."""
    bank = make_bank(min_usage=0.05, min_age=50)

    bank.active_mask[0] = True
    bank.usage_ema[0]   = 0.0    # zero usage
    bank.slot_age[0]    = 10     # young — protected

    bank.prune()

    assert bank.active_mask[0], "Young slot must not be pruned"
    assert bank.active_count == 1


# ------------------------------------------------------------------
# test_merge_combines_similar_slots
# ------------------------------------------------------------------

def test_merge_combines_similar_slots():
    """Two slots with cosine similarity > merge_threshold should be merged."""
    bank = make_bank(merge_threshold=0.95)

    # Two nearly-identical slots
    base_key = F.normalize(torch.ones(D_KEY), dim=-1)
    bank.active_mask[0] = True
    bank.active_mask[1] = True
    bank.slots_key[0]   = base_key.clone()
    bank.slots_key[1]   = base_key.clone()  # identical -> sim = 1.0
    bank.slots_value[0] = torch.zeros(D_VAL)
    bank.slots_value[1] = torch.ones(D_VAL)
    bank.usage_ema[:2]  = 0.1

    bank.merge()

    assert bank.active_count == 1, \
        f"Expected 1 slot after merge, got {bank.active_count}"
    # Merged value should be the mean
    expected_val = torch.ones(D_VAL) * 0.5
    surviving_idx = bank.active_mask.nonzero(as_tuple=True)[0][0].item()
    merged_val = bank.slots_value[surviving_idx]
    assert torch.allclose(merged_val, expected_val, atol=1e-5), \
        f"Merged value mismatch. Got {merged_val[:4]}"


# ------------------------------------------------------------------
# test_merge_keeps_dissimilar_slots
# ------------------------------------------------------------------

def test_merge_keeps_dissimilar_slots():
    """Slots with low cosine similarity must NOT be merged."""
    bank = make_bank(merge_threshold=0.95)

    bank.active_mask[0] = True
    bank.active_mask[1] = True
    # Orthogonal keys -> similarity = 0
    k0 = torch.zeros(D_KEY); k0[0] = 1.0
    k1 = torch.zeros(D_KEY); k1[1] = 1.0
    bank.slots_key[0] = k0
    bank.slots_key[1] = k1
    bank.usage_ema[:2] = 0.1

    bank.merge()

    assert bank.active_count == 2, \
        f"Expected 2 dissimilar slots to remain, got {bank.active_count}"


# ------------------------------------------------------------------
# test_active_count_never_exceeds_max
# ------------------------------------------------------------------

def test_active_count_never_exceeds_max():
    """Repeated high-error writes must never exceed max_slots."""
    max_slots = 5
    bank = make_bank(max_slots=max_slots, error_threshold=0.0)
    bank.running_mean_loss = torch.tensor(1.0)

    for _ in range(20):
        hidden = make_hidden()
        out = bank.read(hidden)
        target = torch.randn(4, D_VAL)
        bank.write(out["query"], target, current_loss=999.0, attn=out["attn"])

    assert bank.active_count <= max_slots, \
        f"Active slots {bank.active_count} exceeded max_slots {max_slots}"


# ------------------------------------------------------------------
# test_no_nan_in_attention_weights
# ------------------------------------------------------------------

def test_no_nan_in_attention_weights():
    """Attention weights must never contain NaN, including with zero slots."""
    bank = make_bank()

    # Case 1: no active slots
    hidden = make_hidden()
    out = bank.read(hidden)
    assert not torch.isnan(out["ctx"]).any(), "NaN in ctx with 0 active slots"

    # Case 2: with active slots
    bank.active_mask[:3] = True
    bank.slots_key[:3] = torch.randn(3, D_KEY)
    out2 = bank.read(hidden)
    assert not torch.isnan(out2["ctx"]).any(), "NaN in ctx with active slots"
    assert not torch.isnan(out2["attn"]).any(), "NaN in attn weights"


# ------------------------------------------------------------------
# test_usage_ema_decays_correctly
# ------------------------------------------------------------------

def test_usage_ema_decays_correctly():
    """After a read with attention 1.0, usage_ema should update via EMA formula."""
    bank = make_bank(usage_ema_decay=0.9)
    bank.active_mask[0] = True
    bank.slots_key[0]   = bank.key_proj.weight.T[:D_KEY, 0]  # Any key
    bank.usage_ema[0]   = 0.0

    hidden = make_hidden(1)

    # Override key to guarantee this slot gets all attention
    query = bank.key_proj(hidden)  # [1, D_KEY]
    bank.slots_key[0] = F.normalize(query.detach().mean(0), dim=-1)

    out = bank.read(hidden)
    attn = out["attn"]  # [1, 1] -> should be ~1.0 (only one slot)

    if attn.numel() > 0:
        expected_usage = (1 - 0.9) * attn.mean(0)[0].item()
        actual_usage   = bank.usage_ema[0].item()
        assert abs(actual_usage - expected_usage) < 0.05, \
            f"Usage EMA mismatch: expected ~{expected_usage:.4f}, got {actual_usage:.4f}"


# ------------------------------------------------------------------
# test_read_output_shape
# ------------------------------------------------------------------

def test_read_output_shape():
    """Read output shapes must match specification."""
    bank = make_bank()
    bank.active_mask[:5] = True
    bank.slots_key[:5] = torch.randn(5, D_KEY)

    batch = 8
    hidden = make_hidden(batch)
    out = bank.read(hidden)

    assert out["ctx"].shape   == (batch, D_VAL), f"ctx shape: {out['ctx'].shape}"
    assert out["attn"].shape  == (batch, 5),     f"attn shape: {out['attn'].shape}"
    assert out["query"].shape == (batch, D_KEY),  f"query shape: {out['query'].shape}"


# ------------------------------------------------------------------
# test_growth_disabled_flag
# ------------------------------------------------------------------

def test_growth_disabled_flag():
    """With growth_disabled=True, no new slots should be activated on high error."""
    bank = make_bank()
    bank.growth_disabled = True
    bank.running_mean_loss = torch.tensor(1.0)

    # Activate one slot so write has something to soft-update
    bank.active_mask[0] = True
    bank.slots_key[0] = F.normalize(torch.ones(D_KEY), dim=-1)

    initial_count = bank.active_count
    hidden = make_hidden()
    out = bank.read(hidden)
    bank.write(out["query"], torch.randn(4, D_VAL), current_loss=999.0, attn=out["attn"])

    assert bank.active_count == initial_count, \
        "Slot count should not change when growth is disabled"
