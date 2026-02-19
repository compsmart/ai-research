"""
Stability Tests — Called In-Loop During Training
-------------------------------------------------
These assertions are called every eval period during training to catch
divergence, NaN propagation, and slot bound violations early.

Also used as standalone tests:
    pytest tests/test_stability.py -v
"""

import pytest
import torch
import torch.nn as nn
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.memory_bank import MemoryBank
from training.metrics import compute_grad_norm, has_nan_gradients


# ------------------------------------------------------------------
# Stability assertion functions (called from trainer)
# ------------------------------------------------------------------

def assert_no_nan_in_gradients(model: nn.Module) -> None:
    """Raise AssertionError if any gradient in the model is NaN."""
    for name, p in model.named_parameters():
        if p.grad is not None and torch.isnan(p.grad).any():
            raise AssertionError(f"NaN gradient detected in {name}")


def assert_gradient_norm_below(model: nn.Module, threshold: float = 10.0) -> None:
    """Raise AssertionError if grad norm exceeds threshold."""
    norm = compute_grad_norm(model)
    if norm > threshold:
        raise AssertionError(
            f"Gradient norm {norm:.4f} exceeds threshold {threshold}"
        )


def assert_loss_not_diverging(loss_history, window: int = 20,
                               max_relative_increase: float = 2.0) -> None:
    """
    Raise AssertionError if loss has increased by more than max_relative_increase
    times its earlier mean within the last `window` steps.
    """
    if len(loss_history) < window:
        return
    half = window // 2
    recent = list(loss_history)[-window:]
    prev_mean = sum(recent[:half]) / half
    last_mean = sum(recent[half:]) / half
    if prev_mean > 0 and last_mean > prev_mean * max_relative_increase:
        raise AssertionError(
            f"Loss appears to be diverging: prev_mean={prev_mean:.4f}, "
            f"last_mean={last_mean:.4f} (>{max_relative_increase}x)"
        )


def assert_slot_count_bounded(memory_bank: MemoryBank, max_slots: int) -> None:
    """Raise AssertionError if active slot count exceeds max_slots."""
    n = memory_bank.active_count
    if n > max_slots:
        raise AssertionError(
            f"Active slot count {n} exceeds max_slots {max_slots}"
        )


def assert_slot_count_not_zero(memory_bank: MemoryBank) -> None:
    """Raise AssertionError if there are zero active slots (memory unused)."""
    n = memory_bank.active_count
    if n == 0:
        raise AssertionError(
            "Memory bank has zero active slots — memory is not being used"
        )


# ------------------------------------------------------------------
# Pytest tests for the assertion functions themselves
# ------------------------------------------------------------------

def test_assert_no_nan_in_gradients_passes_clean():
    model = nn.Linear(4, 4)
    x = torch.randn(2, 4)
    loss = model(x).sum()
    loss.backward()
    assert_no_nan_in_gradients(model)  # Should not raise


def test_assert_no_nan_in_gradients_detects_nan():
    model = nn.Linear(4, 4)
    x = torch.randn(2, 4)
    loss = model(x).sum()
    loss.backward()
    # Inject NaN into the first element of the first gradient tensor (works for any shape)
    for p in model.parameters():
        if p.grad is not None:
            p.grad.flatten()[0] = float("nan")
            break
    with pytest.raises(AssertionError, match="NaN gradient"):
        assert_no_nan_in_gradients(model)


def test_assert_gradient_norm_below_passes():
    model = nn.Linear(4, 4)
    x = torch.randn(2, 4)
    loss = model(x).sum()
    loss.backward()
    # Grad norm of a small model with randn input is typically < 10
    assert_gradient_norm_below(model, threshold=1000.0)


def test_assert_gradient_norm_below_fails_on_large_norm():
    model = nn.Linear(4, 4)
    # Manually set huge gradients
    for p in model.parameters():
        p.grad = torch.full_like(p, fill_value=1000.0)
    with pytest.raises(AssertionError, match="Gradient norm"):
        assert_gradient_norm_below(model, threshold=1.0)


def test_assert_loss_not_diverging_passes_on_stable():
    history = [1.0] * 40
    assert_loss_not_diverging(history, window=20)  # Should not raise


def test_assert_loss_not_diverging_detects_explosion():
    # The last `window` entries must straddle the transition for detection.
    # First half of window=20: [1.0]*10, second half: [100.0]*10
    history = [1.0] * 10 + [100.0] * 10
    with pytest.raises(AssertionError, match="diverging"):
        assert_loss_not_diverging(history, window=20, max_relative_increase=2.0)


def test_assert_slot_count_bounded_passes():
    bank = MemoryBank(max_slots=10, hidden_dim=64)
    bank.active_mask[:5] = True
    assert_slot_count_bounded(bank, max_slots=10)


def test_assert_slot_count_bounded_fails():
    bank = MemoryBank(max_slots=5, hidden_dim=64)
    bank.active_mask[:5] = True
    # Simulate count exceeding max (shouldn't happen normally, but test assertion)
    with pytest.raises(AssertionError, match="max_slots"):
        assert_slot_count_bounded(bank, max_slots=4)


def test_assert_slot_count_not_zero_passes():
    bank = MemoryBank(max_slots=10, hidden_dim=64)
    bank.active_mask[0] = True
    assert_slot_count_not_zero(bank)


def test_assert_slot_count_not_zero_fails():
    bank = MemoryBank(max_slots=10, hidden_dim=64)
    with pytest.raises(AssertionError, match="zero active slots"):
        assert_slot_count_not_zero(bank)
