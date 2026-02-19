# Sprint 02 Status
# Last updated: 2026-02-18

## Completed
- [x] CurriculumRecall task (fixed K concepts, partial cue per episode)
- [x] MemoryBankV2 (novelty trigger using max_cos, not post-softmax attention)
- [x] AdaptiveModelV2 (concept-keyed memory, NOT LSTM-hidden-keyed)
- [x] TrainerV2 (passes max_cos from read_out to write())
- [x] 19 unit tests passing
- [x] Smoke test: K=4 (4 slots), K=8 (7 slots), K=16 (16 slots), all 100% acc
- [x] run_curriculum.py path ordering fixed (sprint02 at sys.path[0])

## In Progress
- [ ] Full diagnostic: fixed_lstm vs ntm_lite vs adaptive, K=4,8,16,32, 3 seeds
  Command: cd sprint02/ && python experiments/run_curriculum.py --K 4 8 16 32 --seeds 3
  Config: config/sprint02.yaml (max_steps=10000 — consider reducing to 5000 for speed)

## Next Steps (in order)
1. Complete full diagnostic run, record acc mean/std per model per K
2. Check sample efficiency: plot accuracy curves, find step where each model first hits 90%
3. Gate 3 check: late_slot_change < 0.10 for adaptive
4. Slot specialization analysis:
   cd sprint02/ && python analysis/slot_specialization.py --exp curriculum_adaptive_K8 --seed 0
5. Extrapolation experiment: train on K=4,8,16 only, eval on K=32 fresh batch (no retraining)
6. If AMM outperforms at K=16/32 → run 5-seed stats (Wilcoxon test)
7. Write Sprint 02 outcome report → sprint02/RESULTS.md

## Key Hypothesis
Slot count at convergence ≈ K (number of fixed concepts)
AMM should learn K distinct memory slots, one per concept.
After K slots are learned, max_cos ≥ 0.5 for any concept query → growth stops.

## Sprint 03 Direction (conditional)
- If Level 2 confirmed (AMM > baselines at K=16,32 with p<0.05):
  Sprint 03: Generalization + transfer. Train K=4-16, test K=32,64.
- If Level 1 only (AMM ≈ baselines, slots just scale):
  Sprint 03: Targeted analysis. What does memory contribute?
  Test: zero top-K slots → accuracy drop = memory contribution score.

## Architecture Notes
- AdaptiveModelV2.forward() extracts concept_query = inputs[:, -1, :self.key_dim]
- This one-hot vector goes through key_proj (key_dim -> d_key) for slot addressing
- LSTM hidden state goes through output_head (hidden_dim + d_val -> output_dim)
- Memory slots store target one-hots as values (d_val = output_dim = 32)

## Results So Far (smoke test, 1 seed, 3K steps)
| K  | max_slots | acc  | slot_curve                                    |
|----|-----------|------|-----------------------------------------------|
|  4 |         4 | 1.00 | [4,4,4,4,4,4,4,4,4,4]                         |
|  8 |         7 | 1.00 | [6,7,7,7,7,7,7,7,7,7]                         |
| 16 |        18 | 1.00 | [18,16,16,16,16,16,16,16,16,16]               |
