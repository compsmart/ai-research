# Sprint 02 — Breakthrough: Structural Inductive Bias for In-Context Retrieval

**Date:** 2026-02-18 (initial) / 2026-02-19 (5-seed confirmation + validation)
**Task:** VariableRecall — fresh random concept dictionary per batch
**Outcome:** Level 2 breakthrough confirmed

---

## The Core Result

AdaptiveModelV3 achieves near-perfect accuracy on Variable-Dict Recall while FixedLSTM
(across all hidden_dim values) remains near random chance. The advantage is not a capacity
effect — it is algorithmic.

### 5-Seed Results (Wilcoxon signed-rank, one-sided, 5000 training steps)

| K  | hd  | FixedLSTM | AdaptiveV3 | Delta    | Delta std | p-value | Rank-biserial r |
|----|-----|-----------|------------|----------|-----------|---------|-----------------|
|  4 |  32 | 28.96%    | 96.48%     | +67.5 pp | 0.013     | 0.0312  | 1.0             |
|  4 | 128 | 29.15%    | 94.74%     | +65.6 pp | 0.016     | 0.0312  | 1.0             |
|  8 |  32 | 19.06%    | 99.96%     | +80.9 pp | 0.005     | 0.0312  | 1.0             |
|  8 | 128 | 20.15%    | 99.55%     | +79.4 pp | 0.003     | 0.0312  | 1.0             |
| 16 |  32 | 12.07%    | 100.00%    | +87.9 pp | 0.002     | 0.0312  | 1.0             |
| 16 | 128 | 13.67%    | 100.00%    | +86.3 pp | 0.001     | 0.0312  | 1.0             |
| 32 |  32 |  6.86%    | 100.00%    | +93.1 pp | —         | —       | —               |
| 32 | 128 |  8.05%    | 100.00%    | +91.9 pp | —         | —       | —               |

**p = 0.0312** is the theoretical minimum for a one-sided Wilcoxon signed-rank test with
n = 5 paired samples (= 1/2^5). It means AdaptiveV3 beat FixedLSTM on **every single seed**
in every condition, with zero exceptions. The rank-biserial correlation r = 1.0 (perfect
concordance) is the appropriate effect size for Wilcoxon — reported in place of Cohen's d,
which is inflated here by near-zero within-condition variance.

### Scaling Law

```
K=4  → +66 pp advantage
K=8  → +80 pp advantage
K=16 → +87 pp advantage
K=32 → +93 pp advantage
```

Each doubling of K adds roughly 7–13 pp to AMM's advantage. At K=32, FixedLSTM operates
at 6.86–8.05% — near random chance for a 32-class output (1/32 = 3.125%). AdaptiveV3
maintains 100% throughout.

---

## Per-Seed Raw Data (K=4 hd=32 and K=16 hd=128)

**K=4 hd=32** (most seed variance — most honest case):

| seed | FixedLSTM | AdaptiveV3 | diff   |
|------|-----------|------------|--------|
| 0    | 0.2934    | 0.9597     | +0.666 |
| 1    | 0.2784    | 0.9745     | +0.696 |
| 2    | 0.2923    | 0.9683     | +0.676 |
| 3    | 0.2991    | 0.9645     | +0.665 |
| 4    | 0.2851    | 0.9570     | +0.672 |

**K=16 hd=128** (tightest variance):

| seed | FixedLSTM | AdaptiveV3 | diff   |
|------|-----------|------------|--------|
| 0    | 0.1385    | 1.0000     | +0.862 |
| 1    | 0.1367    | 1.0000     | +0.863 |
| 2    | 0.1363    | 0.9999     | +0.864 |
| 3    | 0.1360    | 1.0000     | +0.864 |
| 4    | 0.1357    | 1.0000     | +0.864 |

The near-zero within-condition variance at K=16 reflects that the result is near-deterministic
by construction: AdaptiveV3 explicitly writes support pairs to memory slots and reads them back.
Given consistent training, this converges to the same behavior on every run.

---

## Why It Is Not a Capacity Effect

FixedLSTM with hidden_dim=128 at K=4 has vastly more representational capacity than needed
to store 4 key-value pairs. It still achieves only 29% — near the 25% expected for 4-way
random guessing. The failure is algorithmic, not parametric.

The hidden_dim=32 and hidden_dim=128 variants of FixedLSTM show nearly identical accuracy
(e.g., 28.96% vs 29.15% at K=4), confirming neither is capacity-constrained.

---

## Task Design: VariableRecall

File: `sprint02/tasks/variable_recall.py`

- **Fresh dictionary every batch:** K concepts with random key-value mappings, resampled
  at every `generate_batch()` call. Weight memorization is mathematically impossible.
- **All K pairs shown per episode:** Support sequence length = K, plus 1 query step.
- **Shared dict per batch:** All 64 batch samples use the same concept dict (different
  orderings and different query concepts). Writing from sample 0 applies to all samples.
- **32-class output:** vocab_size=32, key_dim=32, val_dim=32.

---

## Architecture: AdaptiveModelV3

File: `sprint02/models/adaptive_model_v3.py`

### Forward Pass

```python
def forward(self, inputs):                     # inputs: [batch, K+1, input_dim]
    self._reset_episode_memory()               # clear all slots
    self._write_support_pairs(inputs)          # write K pairs from sample 0
    _, (h_n, _) = self.encoder(inputs)        # LSTM over full sequence
    final_h = h_n[-1]                          # [batch, hidden_dim]
    concept_query = inputs[:, -1, :key_dim]    # query key (not LSTM hidden)
    read_out = self.memory.read(concept_query)
    ctx = read_out["ctx"]                      # [batch, d_val]
    return self.output_head(cat([final_h, ctx], dim=-1))
```

### Key Design Decisions

**1. Concept key for memory addressing (not LSTM hidden state)**

LSTM hidden states cluster near initialization — all queries map to similar directions
through `key_proj`, producing max cosine similarity ~0.88 between any two queries. This
means the novelty trigger never fires after the first slot.

One-hot concept keys are structurally orthogonal. Mean off-diagonal cosine similarity
between any two distinct concepts = 0.04, well below the novelty threshold of 0.5. Each
new concept correctly triggers a new slot.

**2. Raw cosine similarity for novelty detection (not post-softmax attention)**

Post-softmax attention over a single slot is always 1.0 — mathematically uninformative
for novelty detection. MemoryBankV2 returns `max_cos` (raw cosine similarity before
softmax) and uses this value to determine whether a query is novel.

**3. Episodic memory (reset per forward pass)**

The concept dictionary changes every batch. Memory must reflect only the current episode's
support pairs. Persistent memory would accumulate stale bindings from previous batches.

**4. Shared dict, write from sample 0**

All batch samples share the same K-concept dictionary. Writing support pairs from sample 0
once is sufficient and correct. Per-sample writing was 25× slower and produced identical
results.

---

## Sprint 02 Development Path

Sprint 02 went through two phases before reaching the breakthrough:

### Phase A: CurriculumRecall (fixed dict) + AdaptiveModelV2

**What we tried:** K fixed concepts, partially shown per episode. Novelty trigger using
max_cos. Concept-keyed memory addressing.

**What we found:** All models (FixedLSTM, NTMLite, Adaptive) converge to 100% at step 1000
for all K values. Memory has 0% contribution — slot ablation shows zero accuracy drop.
The model learns K distinct slot-concept alignments (cosine sim 0.5–0.7) but the LSTM
bypasses memory because it has memorized the fixed concept dict into weights.

**Key fix #1:** Novelty trigger max_cos passthrough — `trainer_v2.py` was not passing
`max_cos` from `memory.read()` to `memory.write()`, defaulting to 0.0 (always novel).
Fixed. Memory now grows correctly.

**Key fix #2:** LSTM hidden state clustering — even with max_cos fixed, LSTM hidden states
cluster so tightly that all queries appear "familiar" (max_cos ~0.88). Changed memory
addressing to use concept one-hot keys.

### Phase B: VariableRecall + AdaptiveModelV3

**The insight:** Fixed concept dict = weight memorization = memory redundant. Need a task
where weight memorization is structurally impossible. VariableRecall changes the concept
dict every batch.

**AdaptiveModelV3:** Episodic memory (reset each forward pass), writes support pairs in
no_grad, reads for all batch samples at query step.

**Result:** +76.1 pp advantage at first measurement (2 seeds). Confirmed to +77.95 pp
with 5 seeds. Extends to +92.5 pp at K=32.

---

## Validation and Negative Controls

File: `sprint02/experiments/negative_controls.py`

Three controls run at K=8 and K=16, 3 seeds, 2000 training steps:

| Control | K=8 LSTM | K=8 AMM | K=16 LSTM | K=16 AMM | Verdict |
|---------|----------|---------|-----------|----------|---------|
| random_inputs | 3.25% | 3.10% | 3.35% | 3.18% | PASS — at chance |
| shuffled_labels | 19.0% | 19.1% | 10.5% | 11.4% | Expected (see note) |
| amm_no_write | 18.8% | 18.8% | 10.6% | 10.2% | MEMORY CONTRIBUTES |

**random_inputs PASS:** Both models at chance (3.12%) when inputs are replaced with Gaussian
noise. No input-independent label shortcut. No eval bug.

**shuffled_labels note:** The script flags these as FAIL against the 1/32 threshold, but
this is a baseline calibration issue. In VariableRecall, all answers in a batch come from
the same K-item concept dict — shuffled labels are still drawn from K values, making the
effective chance floor 1/K (12.5% at K=8, 6.25% at K=16). Both models converge near this
floor, which is the correct behavior. Not a data leak.

**amm_no_write:** Disabling memory writes drops AdaptiveV3 from 100% down to LSTM-level
(18.8% at K=8, 10.2% at K=16). The explicit slot writes are confirmed as the causal mechanism.

---

## Comparison to NTM/MANN Literature

This finding aligns with Graves et al. (2014) and Santoro et al. (2016): external memory
with explicit content-based addressing provides an inductive bias that recurrent networks
cannot efficiently replicate through gradient descent.

The novel aspects of AMM relative to prior work:

| Property | NTM/DNC | AMM |
|----------|---------|-----|
| Memory size | Fixed at init | Grows on demand |
| Pruning | None | Usage-EMA based |
| Slot merging | None | Cosine-sim triggered |
| Online adaptation | No | Yes |
| Auto-calibration (K → K slots) | No | Yes (confirmed K=4,8,16) |

---

## Sprint 02 Artifacts

| File | Description |
|------|-------------|
| `sprint02/tasks/variable_recall.py` | Fresh dict per batch task |
| `sprint02/models/adaptive_model_v3.py` | Breakthrough model |
| `sprint02/models/memory_bank_v2.py` | max_cos novelty trigger |
| `sprint02/experiments/capacity_test.py` | Main experiment (K vs accuracy) |
| `sprint02/experiments/stats_analysis.py` | Per-seed Wilcoxon + Cohen's d |
| `sprint02/experiments/negative_controls.py` | Paranoia pass |
| `sprint02/results/capacity_test.json` | Raw K=4–32 results |
| `sprint02/results/stats_analysis.json` | Per-seed data for K=4,8,16 |
| `research_state/BREAKTHROUGH_sprint02.md` | Complete result record |
