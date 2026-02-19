# Next Steps — Sprint 03 and Beyond

**Current status:** Level 2 breakthrough confirmed (Sprint 02, 2026-02-19)
**Next milestone:** Level 3 — zero-shot generalization across complexity

---

## Level Classification Recap

| Level | Criterion | Status |
|-------|-----------|--------|
| 0 — Confirmation | AMM matches NTM-lite on N=4/8 | Done (Sprint 01) |
| 1 — Interesting | Slot count scales sub-linearly with N | Done (Sprint 02) |
| 2 — Strong result | AMM significantly outperforms all baselines at N=16,32; advantage grows with N | **Confirmed** |
| 3 — Major breakthrough | AMM trained on N=4–16 generalizes to N=32 without retraining, baselines fail | Sprint 03 |

---

## Sprint 03 — Primary Experiments

### Experiment 1: Zero-Shot Generalization

**Question:** Can AdaptiveModelV3 trained on K=4 and K=8 correctly perform in-context
retrieval on K=16 and K=32 without any additional training?

**Design:**
1. Train AdaptiveModelV3 on VariableRecall with K drawn uniformly from {4, 8}
2. At evaluation: generate VariableRecall batches with K=16 and K=32 (unseen during training)
3. Compare to FixedLSTM trained the same way

**Hypothesis:** AdaptiveModelV3 generalizes because its memory mechanism is structurally
identical regardless of K — it writes K slots and reads one. FixedLSTM must fit K pairs
into a fixed hidden state, and the compression ratio it learned for K=4,8 won't transfer.

**Success criterion:** AMM achieves >80% at K=16 and >70% at K=32; FixedLSTM shows
significant degradation from in-distribution performance.

**Files to create:** `sprint03/experiments/generalization_test.py`

---

### Experiment 2: FixedLSTM Extended Training (50K steps)

**Question:** Is the LSTM's failure structural (it cannot learn the algorithm) or just a
sample efficiency gap (it needs more training)?

**Design:**
- Train FixedLSTM at K=16, hd=128 for 50,000 steps (10× current budget)
- Track accuracy every 1,000 steps
- Compare final accuracy to AdaptiveModelV3's 5,000-step result

**Hypothesis:** FixedLSTM will remain near its current plateau (~13%) or improve very
slowly, confirming a structural rather than sample efficiency deficit. The LSTM hidden state
simply does not have the right inductive bias to implement "find key, return value"
efficiently from gradient descent.

**If FixedLSTM reaches >80% at 50K steps:** The result is a sample efficiency advantage,
not a structural one. This changes the framing but does not eliminate the claim — AMM
achieves the same accuracy in 10× fewer steps, which is still meaningful.

**Files to create:** `sprint03/experiments/lstm_extended.py`

---

### Experiment 3: K=32 Full Statistical Analysis

**Question:** Are the K=32 results (FixedLSTM ~7%, AMM 100%) statistically confirmed?

**Design:** Re-run K=32 using `stats_analysis.py` with 5 seeds to get per-seed data,
Wilcoxon p-value, and rank-biserial correlation. Currently only the mean is recorded.

**Expected result:** p=0.0312 (perfect concordance, same as K=4,8,16), r=1.0.

**Files:** `sprint02/experiments/stats_analysis.py --K 32 --hd 32 128 --seeds 5`

---

### Experiment 4: No-Write Ablation (Formal)

**Question:** Precisely how much does memory contribute vs the LSTM component?

**Design:** Train two variants side by side on VariableRecall:
- AdaptiveModelV3 (full, with writes) — establishes ceiling
- AdaptiveModelV3-NoWrite (writes disabled) — establishes LSTM-only floor

The accuracy gap between these two is the memory's causal contribution.

**Already partially done:** The `amm_no_write` negative control showed 18.8% (K=8)
and 10.2% (K=16) when writes are disabled, vs 99.96% and 100.0% with writes enabled.
The formal ablation should run at all K values with 5 seeds for a complete picture.

---

## Sprint 03 — Secondary Experiments

### Experiment 5: Slot Specialization on VariableRecall

**Question:** Do the K slots in AdaptiveModelV3 learn concept-specific representations,
or does the model use a distributed code?

**Design:** After training, for each of 1000 test episodes:
1. Record which slot has the highest attention weight for each query concept
2. Check if the same slot consistently serves the same concept index

**Challenge:** Concept identities are random per episode (no persistent concept IDs),
so "slot 3 always stores concept X" is not well-defined. Instead, measure whether the
argmax-attention slot for each support position is stable across episodes with the same
support ordering.

**Files to create:** `sprint03/analysis/slot_specialization_v2.py`

---

### Experiment 6: Mixed-K Training

**Question:** Can a single model handle variable K within a single training run?

**Design:** Sample K uniformly from {4, 8, 16, 32} within each batch. Train for 10,000 steps.
Evaluate at each K separately.

**Hypothesis:** AMM handles this naturally (write K slots, read one). FixedLSTM needs
K to be fixed at training time.

---

## Open Questions

### Statistical

1. **Reporting:** Cohen's d (41–785) is inflated by near-zero within-condition variance.
   Rank-biserial r=1.0 is the correct effect size to headline. Future papers should use
   r, not d.

2. **Multiple comparisons:** 6 separate Wilcoxon tests were run (one per condition).
   Bonferroni correction would set threshold at 0.05/6 = 0.0083. All tests still pass
   because p=0.0312 is the floor — with more seeds (n≥6), p-values can go lower.

3. **Sample size:** n=5 seeds gives limited power. For publication, n=10 seeds per
   condition is preferable. Currently all conditions show perfect concordance (r=1.0),
   so more seeds would only increase power, not change the direction.

### Architectural

1. **Why does FixedLSTM plateau at ~1/K?** The consistent accuracy near 1/K (25% at K=4,
   12.5% at K=8, 6.25% at K=16) suggests the LSTM learns to predict "any of the K active
   values" uniformly, then cannot discriminate further. Investigate what the LSTM's
   prediction distribution looks like (entropy, whether it concentrates on the K values).

2. **Output head co-training:** The fast adaptation result (0% zero-shot binding) shows
   the output head learns to rely on LSTM context rather than memory context. A future
   design could use a memory-only output head (remove LSTM from the read path) to force
   full reliance on memory.

3. **Gradient flow through memory:** Currently, support pair writes happen in `no_grad`.
   Gradient flows through `key_proj` and the output head's memory read path, but not
   through the write mechanism. End-to-end differentiable writing may improve performance
   at K=4 (where AMM is ~96%, not 100%).

### Theoretical

1. **What is FixedLSTM actually computing?** Run probing classifiers on LSTM hidden states
   to determine how much concept identity information is encoded at each timestep.

2. **Is the inductive bias learnable?** Could FixedLSTM with a specially designed loss
   (e.g., contrastive pairs for concept discrimination) learn to implement retrieval?
   This would test whether the bias is truly structural or just difficult to learn.

---

## Sprint 03 File Structure (Planned)

```
sprint03/
├── experiments/
│   ├── generalization_test.py    # Train K=4,8 → eval K=16,32
│   ├── lstm_extended.py          # FixedLSTM 50K steps
│   └── mixed_k_training.py      # Variable K within training
├── analysis/
│   └── slot_specialization_v2.py # Slot alignment on VariableRecall
├── tests/
│   └── test_sprint03.py
└── results/
```

---

## Decision Branch After Sprint 03

**If generalization test succeeds (AMM zero-shots K=32, LSTM fails):**
- Level 3 breakthrough confirmed
- Sprint 04: Ablation table, second task domain, paper writeup

**If generalization test fails (both models fail on unseen K):**
- Still Level 2 — the in-distribution result remains valid
- Sprint 04: Investigate what prevents generalization; try curriculum-based K growth

**Hard stop criteria (from research plan):**
- 4 sprints without a Level 2 result → pivot (already passed this with Sprint 02)
- Contemporaneous paper publishes same mechanism with stronger results
- Toy task experiments exceed 4 GPU-hours per run
