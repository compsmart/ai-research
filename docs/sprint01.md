# Sprint 01 — Feasibility Study

**Date:** Early February 2026
**Task:** AssociativeRecall with fixed concept dictionary
**Outcome:** Branch B3 — memory ignored

---

## Objective

Establish whether an adaptive memory bank with error-triggered slot growth and
usage-based pruning could outperform a fixed LSTM on an associative recall task.
The primary hypothesis: slot count at convergence should scale with task difficulty N
(number of key-value pairs to recall).

---

## Task Design: AssociativeRecall

- N (key, value) pairs presented as context, followed by a query key
- Model must return the correct value
- Difficulty ladder: N = 4, 8, 16, 32
- Fixed concept dictionary: same K concepts reused across all episodes and batches

---

## Architecture

### MemoryBank (sprint01/models/memory_bank.py)

Pre-allocated slot buffer with activation mask (avoids mid-training optimizer state issues):

```
slots_key:    Tensor[max_slots, d_key=32]
slots_value:  Tensor[max_slots, d_val=64]
active_mask:  bool[max_slots]
usage_ema:    float[max_slots]      — EMA of attention received
slot_age:     int[max_slots]        — steps since activation
```

Write trigger: activate new slot when `current_loss > threshold * running_mean_loss`
Prune trigger: deactivate slots below `min_usage` after `min_age` steps
Merge trigger: merge slot pairs with cosine similarity > `merge_threshold`

### AdaptiveModel (sprint01/models/adaptive_model.py)

- BaseNet (2-layer MLP, 64 hidden) processes input
- Memory read via cosine attention over active slots
- Output: concat(MLP hidden, memory context) → prediction head

---

## Baselines

| Model | Description |
|-------|-------------|
| Fixed MLP | 2-layer MLP, no memory |
| NTM-lite | 20 pre-allocated slots, fixed, no growth |
| AdaptiveModel | Full AMM with growth + pruning + merge |

---

## Results

**Key finding:** All three models converge to 100% accuracy on the fixed-concept
AssociativeRecall task by step 1000, regardless of N.

| Model | N=4 acc | N=8 acc | N=16 acc | N=32 acc | Steps to 90% |
|-------|---------|---------|----------|----------|--------------|
| FixedLSTM | 100% | 100% | 100% | 100% | ~1000 |
| NTM-lite | 100% | 100% | 100% | 100% | ~1000 |
| Adaptive | 100% | 100% | 100% | 100% | ~1000 |

Memory slot count: NTM-lite stable at 2 slots; Adaptive grew to 24 then pruned to ~22.
Memory is functionally redundant — slot ablation shows 0% accuracy drop when any slot
is removed.

---

## Root Cause: Weight Memorization

The failure was not in the architecture but in the task design. A **fixed concept
dictionary** allows gradient descent to memorize concept-to-value mappings directly
into LSTM weights. Once those weights encode the K concepts, the memory bank
contributes nothing — the LSTM's output head learns to ignore the memory context.

Slot ablation confirmed this: removing any slot had zero effect on accuracy across
all concepts and all N values. The slots aligned geometrically to concepts (cosine
similarity 0.5–0.7 per slot-concept pair), indicating the memory DID learn structure,
but the LSTM had already encoded the same information more reliably.

---

## Diagnosis: Branch B3

Per the research plan's decision tree:

> **B3 (memory ignored):** check attention entropy; force key diversity via regularization

The loss-relative write trigger fired approximately 50% of steps indefinitely
(perpetual cycle), never converging to a stable slot configuration. This was a
secondary symptom of the core problem: with fixed concepts, the LSTM alone is
sufficient and memory is never genuinely needed.

---

## Sprint 01 Artifacts

```
sprint01/
├── models/memory_bank.py         — v1 memory bank (loss-relative trigger)
├── models/adaptive_model.py      — v1 adaptive model
├── tasks/associative_recall.py   — fixed concept dictionary task
├── tests/test_unit.py            — 9 unit tests (all passing)
├── results/diag2/summary.json    — full diagnostic results
└── requirements.txt
```

---

## Lessons Learned

1. **Fixed concept dict = memory redundant.** Gradient descent will always bypass
   explicit memory when it can memorize the same information into weights. Memory
   is only genuinely needed when the task prevents weight-level memorization.

2. **Loss-relative trigger is unstable.** Firing 50% of steps indefinitely provides
   no useful signal about when new memory is actually needed.

3. **Slot alignment ≠ slot utility.** Memory slots can learn meaningful geometric
   structure (aligned to distinct concepts) while being completely bypassed at
   inference time.

---

## Sprint 02 Direction

Redesign the task to make weight memorization impossible:
- Fresh random concept dictionary every batch (VariableRecall)
- Model must learn a general in-context retrieval algorithm
- Memory becomes structurally necessary, not optional

See [sprint02.md](sprint02.md) for results.
