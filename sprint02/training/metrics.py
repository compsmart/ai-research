"""
Metric Computation Helpers
---------------------------
Pure functions that compute metrics from raw data.
"""

import math
import torch
import torch.nn as nn
from typing import List, Optional


def compute_grad_norm(model: nn.Module) -> float:
    """L2 norm of all gradients in the model."""
    total_sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_sq += p.grad.data.norm(2).item() ** 2
    return math.sqrt(total_sq)


def has_nan_gradients(model: nn.Module) -> bool:
    """Return True if any gradient contains NaN."""
    for p in model.parameters():
        if p.grad is not None and torch.isnan(p.grad).any():
            return True
    return False


def has_nan_in_output(tensor: torch.Tensor) -> bool:
    return torch.isnan(tensor).any().item()


def compute_loss_trend(loss_history: List[float], window: int = 20) -> float:
    """
    Return the relative increase in mean loss between the last window/2 and
    the previous window/2. Positive = diverging, negative = converging.
    """
    if len(loss_history) < window:
        return 0.0
    half = window // 2
    prev = sum(loss_history[-window:-half]) / half
    last = sum(loss_history[-half:]) / half
    if prev == 0:
        return 0.0
    return (last - prev) / abs(prev)


def compute_accuracy_from_logits(logits: torch.Tensor,
                                  targets: torch.Tensor) -> float:
    """
    Compute per-sample accuracy.
    logits:  [batch, n_classes] or [batch, seq, n_classes]
    targets: [batch, n_classes] one-hot or [batch, seq, n_classes] one-hot
    """
    if logits.dim() == 3:
        pred = logits.argmax(dim=-1)   # [batch, seq]
        true = targets.argmax(dim=-1)
        return (pred == true).float().mean().item()
    pred = logits.argmax(dim=-1)       # [batch]
    true = targets.argmax(dim=-1)
    return (pred == true).float().mean().item()


def coefficient_of_variation(values: List[float]) -> float:
    """CV = std / mean. Returns 0 if mean is 0."""
    if not values:
        return 0.0
    import statistics
    if len(values) == 1:
        return 0.0
    m = statistics.mean(values)
    if m == 0:
        return 0.0
    return statistics.stdev(values) / abs(m)


def steps_to_threshold(accuracy_history: List[float],
                        threshold: float = 0.9) -> Optional[int]:
    """Return the step index at which accuracy first exceeded threshold, or None."""
    for i, acc in enumerate(accuracy_history):
        if acc >= threshold:
            return i
    return None
