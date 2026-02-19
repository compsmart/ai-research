# Adaptive Modular Memory (AMM) Research

Research question: Can error-triggered slot growth and usage-based pruning auto-calibrate
structural memory capacity to task complexity?

## Status

**Sprint 02 — Level 2 Breakthrough Confirmed (2026-02-19)**

AdaptiveModelV3 vs FixedLSTM on the VariableRecall task (fresh random concept dict per
batch, making weight memorization impossible):

| K  | FixedLSTM | AdaptiveV3 | Delta   | Cohen's d | p      |
|----|-----------|------------|---------|-----------|--------|
|  4 | ~29%      | ~96%       | +67 pp  | 41–54     | 0.0312 |
|  8 | ~20%      | ~100%      | +80 pp  | 174–241   | 0.0312 |
| 16 | ~13%      | 100%       | +87 pp  | 415–785   | 0.0312 |
| 32 | ~7%       | 100%       | +93 pp  | —         | —      |

5 seeds, Wilcoxon signed-rank test. p = 0.0312 is the theoretical minimum for n=5.
AMM beat the LSTM baseline on every single seed at every K. The advantage grows
monotonically with K; at K=32 the LSTM is near chance (1/32 = 3.125%).

Full report: [research_state/BREAKTHROUGH_sprint02.md](research_state/BREAKTHROUGH_sprint02.md)

---

## Research Design

### Task: VariableRecall

Each batch receives a freshly sampled random concept dictionary mapping K abstract keys
to K random vectors. The model sees (key, value) support pairs, then must retrieve the
correct value given a query key. Because the dictionary changes every batch, the LSTM
cannot memorize concept-to-value mappings through gradient descent — correct retrieval
requires in-context memory.

### Models

**AdaptiveModelV3** (breakthrough model, `sprint02/models/adaptive_model_v3.py`)

- Episodic memory bank: slots reset before each forward pass
- Support pairs are written into memory slots at steps 0..K-1
- At query step: concept key addresses memory directly (not via LSTM hidden state)
- Output head: concatenate LSTM final hidden state + memory context vector
- Slot growth triggered by novelty (raw cosine similarity threshold)

**FixedLSTM** (`sprint02/models/`) — standard LSTM with no external memory, used as baseline.

### Critical Architecture Decisions

1. Memory is keyed on the **concept query vector**, not the LSTM hidden state.
2. Novelty detection uses **max cosine similarity** (raw, pre-softmax), not attention weights.
3. VariableRecall (not fixed-concept tasks) is required — fixed concepts allow the LSTM
   to bypass memory by memorizing associations in its weights.

---

## Project Structure

```
ai-research/
├── research_state/           # Sprint reports and status
│   ├── BREAKTHROUGH_sprint02.md
│   ├── sprint02_status.md
│   └── RESUME_PROMPT.md
│
├── sprint01/                 # Sprint 01: feasibility study
│   ├── models/               # AdaptiveModel v1, MemoryBank, BaseNet (LSTM)
│   ├── tasks/                # AssociativeRecall, CopyTask
│   ├── experiments/          # Single-run and multi-seed scripts
│   ├── training/             # Trainer, logger, metrics
│   ├── analysis/             # Wilcoxon tests, visualizations
│   ├── tests/
│   ├── results/
│   └── requirements.txt
│
└── sprint02/                 # Sprint 02: variable-dict recall (breakthrough)
    ├── models/               # AdaptiveModelV2/V3, MemoryBankV2
    ├── tasks/                # VariableRecall, CurriculumRecall
    ├── experiments/          # capacity_test.py, stats_analysis.py, ablations
    ├── training/             # TrainerV2
    ├── analysis/             # Slot specialization analysis
    ├── tests/
    └── results/              # capacity_test.json, stats_analysis.json
```

---

## Setup

```bash
pip install -r sprint01/requirements.txt
```

Dependencies: `torch>=2.0.0`, `numpy>=1.24.0`, `scipy`, `pyyaml>=6.0`, `pytest>=7.0.0`

---

## Running Experiments

**Main capacity experiment (reproduces breakthrough results):**

```bash
python sprint02/experiments/capacity_test.py
```

**Per-seed statistical analysis:**

```bash
python sprint02/experiments/stats_analysis.py
```

**Ablation studies:**

```bash
python sprint02/experiments/run_ablation.py
python sprint02/experiments/slot_ablation.py
```

**Tests:**

```bash
pytest sprint01/tests/
pytest sprint02/tests/
```

---

## Results

Raw per-seed data: `sprint02/results/stats_analysis.json`
Main experiment output: `sprint02/results/capacity_test.json`

---

## Pending (Sprint 03)

1. AdaptiveV3-no-write ablation — isolate memory contribution vs LSTM
2. FixedLSTM extended training (50K steps) — structural vs sample efficiency
3. Generalization: train on K=4,8 then evaluate zero-shot on K=16,32
4. K=32 full statistical analysis (5-seed Wilcoxon + Cohen's d)
