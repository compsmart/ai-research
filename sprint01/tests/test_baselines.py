"""
Baseline Regression Tests
--------------------------
Verify that baseline models (FixedMLP, NTMLite, SmallTransformer) can:
  1. Forward-pass without error
  2. Produce outputs of the correct shape
  3. Compute a loss and backprop without NaN

Run with:
    pytest tests/test_baselines.py -v
"""

import pytest
import torch
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.adaptive_model import FixedMLP, FixedLSTM, NTMLite, SmallTransformer, AdaptiveModel
from tasks.associative_recall import AssociativeRecall
from tasks.copy_task import CopyTask


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

BATCH = 8
N = 4          # associative recall pairs
SEQ_LEN = 5   # copy task length
VOCAB = 32    # vocabulary size (must be >= N and <= key_dim, val_dim)
KEY_DIM = 32
VAL_DIM = 32


@pytest.fixture
def ar_task():
    return AssociativeRecall(n=N, vocab_size=VOCAB, key_dim=KEY_DIM, val_dim=VAL_DIM)


@pytest.fixture
def copy_task():
    return CopyTask(seq_len=SEQ_LEN, vocab_size=8)


# ------------------------------------------------------------------
# FixedMLP
# ------------------------------------------------------------------

class TestFixedMLP:
    def test_forward_shape(self, ar_task):
        inputs, targets = ar_task.generate_batch(BATCH)
        seq_len = inputs.shape[1]
        model = FixedMLP(
            input_dim=ar_task.input_dim,
            seq_len=seq_len,
            output_dim=ar_task.output_dim,
        )
        logits = model(inputs)
        assert logits.shape == (BATCH, ar_task.output_dim)

    def test_backward_no_nan(self, ar_task):
        inputs, targets = ar_task.generate_batch(BATCH)
        seq_len = inputs.shape[1]
        model = FixedMLP(
            input_dim=ar_task.input_dim,
            seq_len=seq_len,
            output_dim=ar_task.output_dim,
        )
        logits = model(inputs)
        loss = torch.nn.CrossEntropyLoss()(logits, targets.argmax(dim=-1))
        loss.backward()
        for p in model.parameters():
            if p.grad is not None:
                assert not torch.isnan(p.grad).any(), "NaN gradient in FixedMLP"


class TestFixedLSTM:
    def test_forward_shape(self, ar_task):
        inputs, targets = ar_task.generate_batch(BATCH)
        model = FixedLSTM(input_dim=ar_task.input_dim, output_dim=ar_task.output_dim)
        logits = model(inputs)
        assert logits.shape == (BATCH, ar_task.output_dim)

    def test_backward_no_nan(self, ar_task):
        inputs, targets = ar_task.generate_batch(BATCH)
        model = FixedLSTM(input_dim=ar_task.input_dim, output_dim=ar_task.output_dim)
        logits = model(inputs)
        loss = torch.nn.CrossEntropyLoss()(logits, targets.argmax(dim=-1))
        loss.backward()
        for p in model.parameters():
            if p.grad is not None:
                assert not torch.isnan(p.grad).any(), "NaN gradient in FixedLSTM"


# ------------------------------------------------------------------
# NTMLite
# ------------------------------------------------------------------

class TestNTMLite:
    def test_forward_shape(self, ar_task):
        inputs, targets = ar_task.generate_batch(BATCH)
        model = NTMLite(
            input_dim=ar_task.input_dim,
            output_dim=ar_task.output_dim,
            n_slots=20,
        )
        logits = model(inputs)
        assert logits.shape == (BATCH, ar_task.output_dim)

    def test_all_slots_active(self, ar_task):
        model = NTMLite(input_dim=ar_task.input_dim, output_dim=ar_task.output_dim,
                        n_slots=20)
        assert model.memory.active_count == 20

    def test_growth_disabled(self, ar_task):
        model = NTMLite(input_dim=ar_task.input_dim, output_dim=ar_task.output_dim,
                        n_slots=20)
        assert model.memory.growth_disabled is True

    def test_backward_no_nan(self, ar_task):
        inputs, targets = ar_task.generate_batch(BATCH)
        model = NTMLite(input_dim=ar_task.input_dim, output_dim=ar_task.output_dim)
        logits = model(inputs)
        loss = torch.nn.CrossEntropyLoss()(logits, targets.argmax(dim=-1))
        loss.backward()
        for p in model.parameters():
            if p.grad is not None:
                assert not torch.isnan(p.grad).any(), "NaN gradient in NTMLite"


# ------------------------------------------------------------------
# SmallTransformer
# ------------------------------------------------------------------

class TestSmallTransformer:
    def test_forward_shape(self, ar_task):
        inputs, targets = ar_task.generate_batch(BATCH)
        model = SmallTransformer(
            input_dim=ar_task.input_dim,
            output_dim=ar_task.output_dim,
        )
        logits = model(inputs)
        assert logits.shape == (BATCH, ar_task.output_dim)

    def test_backward_no_nan(self, ar_task):
        inputs, targets = ar_task.generate_batch(BATCH)
        model = SmallTransformer(
            input_dim=ar_task.input_dim,
            output_dim=ar_task.output_dim,
        )
        logits = model(inputs)
        loss = torch.nn.CrossEntropyLoss()(logits, targets.argmax(dim=-1))
        loss.backward()
        for p in model.parameters():
            if p.grad is not None:
                assert not torch.isnan(p.grad).any(), "NaN gradient in SmallTransformer"


# ------------------------------------------------------------------
# AdaptiveModel
# ------------------------------------------------------------------

class TestAdaptiveModel:
    def test_forward_shape(self, ar_task):
        inputs, targets = ar_task.generate_batch(BATCH)
        model = AdaptiveModel(
            input_dim=ar_task.input_dim,
            output_dim=ar_task.output_dim,
        )
        logits = model(inputs)
        assert logits.shape == (BATCH, ar_task.output_dim)

    def test_starts_with_zero_slots(self, ar_task):
        model = AdaptiveModel(
            input_dim=ar_task.input_dim,
            output_dim=ar_task.output_dim,
        )
        assert model.memory.active_count == 0

    def test_backward_no_nan(self, ar_task):
        inputs, targets = ar_task.generate_batch(BATCH)
        model = AdaptiveModel(
            input_dim=ar_task.input_dim,
            output_dim=ar_task.output_dim,
        )
        logits = model(inputs)
        loss = torch.nn.CrossEntropyLoss()(logits, targets.argmax(dim=-1))
        loss.backward()
        for p in model.parameters():
            if p.grad is not None:
                assert not torch.isnan(p.grad).any(), "NaN gradient in AdaptiveModel"


# ------------------------------------------------------------------
# Task generators
# ------------------------------------------------------------------

class TestTasks:
    def test_associative_recall_shape(self, ar_task):
        inputs, targets = ar_task.generate_batch(BATCH)
        assert inputs.shape  == (BATCH, N + 1, ar_task.input_dim), f"Got {inputs.shape}"
        assert targets.shape == (BATCH, ar_task.output_dim), f"Got {targets.shape}"

    def test_associative_recall_target_is_one_hot(self, ar_task):
        _, targets = ar_task.generate_batch(BATCH)
        assert (targets.sum(dim=-1) == 1).all(), "Targets must be one-hot"

    def test_copy_task_shape(self, copy_task):
        inputs, targets = copy_task.generate_batch(BATCH)
        assert inputs.shape  == (BATCH, SEQ_LEN + 1, copy_task.input_dim)
        assert targets.shape == (BATCH, SEQ_LEN, copy_task.output_dim)

    def test_associative_recall_with_n(self, ar_task):
        task8 = ar_task.with_n(8)
        inputs, targets = task8.generate_batch(2)
        assert inputs.shape[1] == 9  # 8 pairs + 1 query

    def test_copy_task_with_len(self, copy_task):
        task20 = copy_task.with_len(20)
        inputs, targets = task20.generate_batch(2)
        assert inputs.shape[1] == 21  # 20 seq + 1 sep
