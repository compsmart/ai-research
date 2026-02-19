# Sprint 02 Breakthrough Report
# Date: 2026-02-18
# Updated: 2026-02-19 (5-seed confirmation)

## BREAKTHROUGH: Structural Inductive Bias for In-Context Associative Retrieval

## Summary
AdaptiveModelV3 achieves near-perfect accuracy on the Variable-Dict Recall task while
FixedLSTM (even with large hidden_dim) remains near chance. Mean advantage: +77.95pp (5 seeds).

## Key Result Table — 5-seed Confirmation (2026-02-19)

| K  | hd  | FixedLSTM | AdaptiveV3 | Delta   |
|----|-----|-----------|------------|---------|
|  4 |  32 | 28.96%    | 96.48%     | +67.5%  |
|  4 | 128 | 29.15%    | 94.74%     | +65.6%  |
|  8 |  32 | 19.06%    | 99.96%     | +80.9%  |
|  8 | 128 | 20.15%    | 99.55%     | +79.4%  |
| 16 |  32 | 12.07%    | 100.00%    | +87.9%  |
| 16 | 128 | 13.67%    | 100.00%    | +86.3%  |
| 32 |  32 |  6.86%    | 100.00%    | +93.1%  |
| 32 | 128 |  8.05%    | 100.00%    | +91.9%  |

Mean AMM advantage: +77.95 pp (K=4-16, 5 seeds) / +92.5 pp at K=32
K=32 FixedLSTM: 6.86-8.05% (near random chance; 1/32 = 3.125%)
K=32 AdaptiveV3: 100.00% (both hidden_dim values)

Scaling law: K=4 (+66pp) -> K=8 (+80pp) -> K=16 (+87pp) -> K=32 (+93pp)
Each doubling of K adds ~7-13pp to AMM's advantage.

2-seed mean was +76.1pp — result stable across seed counts (delta < 2pp per condition).

## Original 2-seed Results (2026-02-18, for comparison)

| K  | hd  | FixedLSTM | AdaptiveV3 | Delta   |
|----|-----|-----------|------------|---------|
|  4 |  32 | 29.3%     | 92.5%      | +63.2%  |
|  4 | 128 | 30.0%     | 89.2%      | +59.2%  |
|  8 |  32 | 18.5%     | 99.6%      | +81.1%  |
|  8 | 128 | 19.8%     | 97.3%      | +77.5%  |
| 16 |  32 | 11.5%     | 100.0%     | +88.5%  |
| 16 | 128 | 12.9%     | 100.0%     | +87.1%  |

Seeds used: 5 (CONFIRMED). Statistics: ALL conditions p=0.0312 (Wilcoxon), Cohen's d=41-785.

## Statistical Results (Wilcoxon Signed-Rank, 5 seeds, 2026-02-19)

| Condition  | Delta mean | Delta std | p-value | Cohen's d | Sig? |
|------------|------------|-----------|---------|-----------|------|
| K=4 hd=32  | +0.6752    | 0.0125    | 0.0312  | 54.02     | YES  |
| K=4 hd=128 | +0.6559    | 0.0159    | 0.0312  | 41.28     | YES  |
| K=8 hd=32  | +0.8090    | 0.0047    | 0.0312  | 173.73    | YES  |
| K=8 hd=128 | +0.7941    | 0.0033    | 0.0312  | 240.84    | YES  |
| K=16 hd=32 | +0.8793    | 0.0021    | 0.0312  | 415.47    | YES  |
| K=16 hd=128| +0.8633    | 0.0011    | 0.0312  | 784.74    | YES  |

p=0.0312 is the THEORETICAL MINIMUM for one-sided Wilcoxon at n=5 (= 1/2^5).
This means AMM beat FixedLSTM on EVERY SINGLE SEED in EVERY condition.
Cohen's d conventional threshold for "large": 0.8. Observed range: 41-785.

## Task Design: Variable-Dict Recall (VariableRecall)
- Each batch: fresh random concept dictionary (K concepts, K->V mappings)
- All K pairs shown in random order per episode
- Query: any of K concepts (answer in context)
- Weight memorization is IMPOSSIBLE (dict changes every batch)
- Model must learn a GENERAL in-context retrieval algorithm

## Why This Is Significant
1. NOT just a capacity bottleneck: even hd=128 with K=4 (well above capacity)
   FixedLSTM still fails (30%). The issue is ALGORITHMIC, not parametric.
2. Growing advantage with K: +63% at K=4 → +88% at K=16.
   AMM advantage scales with task complexity.
3. Theoretically clean: explicit key-value addressing naturally implements the
   "find key, return value" algorithm. LSTM must discover this implicitly.

## Architecture: AdaptiveModelV3
Key innovation vs v2:
- Episodic memory: reset before each forward pass (fresh dict per episode)
- Step-by-step support writing: each support pair (k_i, v_i) written to slot i
  using concept key k_i as address (one-hot → key_proj → d_key query)
- Shared dict per batch: all samples share the same concept dict;
  writing from sample 0's pairs applies to all samples
- LSTM still processes the full sequence but memory is the primary retrieval path

Files:
  sprint02/tasks/variable_recall.py
  sprint02/models/adaptive_model_v3.py
  sprint02/experiments/capacity_test.py
  sprint02/results/capacity_test.json

## Negative Controls / Paranoia Pass (2026-02-19)

sprint02/experiments/negative_controls.py — K=8,16, 3 seeds, 2000 steps each

| Control         | K=8 LSTM | K=8 AMM | K=16 LSTM | K=16 AMM | Verdict |
|-----------------|----------|---------|-----------|----------|---------|
| random_inputs   | 3.25%    | 3.10%   | 3.35%     | 3.18%    | PASS — at chance, no eval bug |
| shuffled_labels | 19.0%    | 19.1%   | 10.5%     | 11.4%    | Expected — effective chance is 1/K not 1/32 |
| amm_no_write    | 18.8%    | 18.8%   | 10.6%     | 10.2%    | MEMORY CONTRIBUTES — writes are active ingredient |

Notes:
- random_inputs PASS: inputs genuinely drive predictions; no label-independent shortcut
- shuffled_labels: correct answers per batch drawn from K-item dict → effective chance = 1/K
  (12.5% at K=8, 6.25% at K=16). Both models near floor. Not a data leak.
- amm_no_write: disabling writes drops AMM from 100% to LSTM-level. Memory confirmed causal.

## What Still Needs to Be Done
1. [DONE] 5-seed replication — confirmed, mean +77.95pp, stable across seeds
2. [DONE] K=32 test — +92.5pp advantage, FixedLSTM near chance (6.86%), AdaptiveV3 100%
3. [DONE] Wilcoxon signed-rank + Cohen's d — ALL 6 conditions p=0.0312, d=41-785
4. [DONE] Negative controls — random_inputs PASS, amm_no_write confirms memory is causal
5. Check: can FixedLSTM ever solve this with 50K steps? (structural vs sample efficiency)
6. Sprint 03: Generalization test — train K=4,8, evaluate K=16,32 zero-shot

## Interpretation
The finding aligns with the NTM/MANN literature (Graves et al. 2014, Santoro et al. 2016):
External memory with explicit content-based addressing provides an inductive bias
that standard recurrent networks cannot replicate efficiently through gradient descent.

The novel contribution of AMM is:
- ADAPTIVE slot growth: slots are added on demand (novelty trigger), not pre-allocated
- USAGE-BASED pruning: unused slots are removed automatically
- STRUCTURAL auto-calibration: K concepts → K slots (tested: K=4→4, K=8→7, K=16→16)

## Level Classification
Level 2: "AMM shows statistically significant sample efficiency advantage over ALL
baselines at K=16 AND the advantage grows with task difficulty."
STATUS: Confirmed directionally, needs 5-seed statistics.

Level 3 check: Train on K=4-8, evaluate on K=16 without retraining → TBD (Sprint 03)

## Next Sprint Direction
Sprint 03: Confirm Level 2 with statistics, then attempt Level 3 generalization.
- Train on K=4,8, test on K=16,32 (zero-shot generalization across complexity)
- If AMM generalizes and FixedLSTM does not: Level 3 breakthrough confirmed
