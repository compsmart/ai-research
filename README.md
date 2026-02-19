# Adaptive Modular Memory (AMM) Research

**Research question:** Can error-triggered slot growth and usage-based pruning auto-calibrate
structural memory capacity to task complexity during training?

**Current status:** Sprint 02 — Level 2 breakthrough confirmed (2026-02-19)

---

## The Core Finding

Standard recurrent networks (LSTMs) cannot learn a general in-context key-value retrieval
algorithm through gradient descent. A model with explicit key-value memory slots can.

On the Variable-Dict Recall task — where a fresh random concept dictionary is generated
every batch, making weight memorization impossible — AdaptiveModelV3 achieves near-perfect
accuracy while FixedLSTM remains near random chance across all tested complexity levels:

| K (concepts) | FixedLSTM | AdaptiveV3 | Advantage | p-value |
|:------------:|:---------:|:----------:|:---------:|:-------:|
|  4           | 28.96%    | 96.48%     | +67.5 pp  | 0.0312  |
|  8           | 19.06%    | 99.96%     | +80.9 pp  | 0.0312  |
| 16           | 12.07%    | 100.00%    | +87.9 pp  | 0.0312  |
| 32           |  6.86%    | 100.00%    | +93.1 pp  | —       |

5 seeds, paired Wilcoxon signed-rank test. p = 0.0312 is the theoretical minimum for n=5
(AMM beat LSTM on every single seed at every condition). Advantage grows monotonically with K;
at K=32, FixedLSTM operates at chance level (1/32 = 3.125%).

The result is not a capacity effect — FixedLSTM with hidden_dim=128 fails just as badly as
hidden_dim=32 at K=4. The problem is algorithmic: LSTM cannot learn the "find key, return
value" procedure from gradient descent alone.

Full details: [docs/sprint02.md](docs/sprint02.md)

---

## How It Works

### Task: Variable-Dict Recall

Each forward pass receives a freshly sampled random concept dictionary: K abstract keys
mapped to K random values. The model sees all K (key, value) support pairs, then must
retrieve the correct value given a query key. Because the dictionary changes every batch,
gradient descent cannot memorize key-to-value associations in network weights — correct
retrieval requires genuine in-context memory.

### Model: AdaptiveModelV3

```
Input sequence: [pair_0, pair_1, ..., pair_{K-1}, query]

Forward pass:
  1. Reset episodic memory (clear all slots)
  2. Write support pairs: for each (key_i, val_i), write val_i to slot addressed by key_i
  3. Run LSTM over full sequence → final hidden state h
  4. Read from memory using query key → context vector c
  5. Output = head(concat(h, c))
```

Key design decisions:
- Memory is keyed on the **concept query vector** (one-hot), not the LSTM hidden state.
  LSTM hidden states cluster near initialization; concept vectors are orthogonal.
- Novelty detection uses **raw cosine similarity** (pre-softmax), not attention weights.
  Post-softmax on a single slot is always 1.0 — useless for novelty detection.
- Variable concept dict is required. Fixed dicts allow LSTM weight memorization, making
  memory redundant. VariableRecall prevents this by changing the dict every batch.

---

## Comparison to Prior Work

| System              | Memory size | Grows?       | Pruning? | Online? |
|---------------------|-------------|--------------|----------|---------|
| NTM (2014)          | Fixed       | No           | No       | No      |
| DNC (2016)          | Fixed       | No           | No       | No      |
| Progressive NN      | Grows       | Yes          | No       | No      |
| **AMM (this work)** | Variable    | Error-triggered | Usage-based | Yes |

The novel contribution: online error-triggered structural growth with usage-based pruning
and slot merging during training, producing models that auto-calibrate capacity to task
complexity (K concepts → K slots, confirmed at K=4,8,16).

---

## Sprint History

| Sprint | Task | Key Finding | Status |
|--------|------|------------|--------|
| 01 | AssociativeRecall (fixed dict) | Branch B3: memory ignored. LSTM alone sufficient when dict is fixed — gradient descent memorizes associations into weights. | Done |
| 02 | VariableRecall (fresh dict/batch) | Level 2 breakthrough: AMM +67–93pp over LSTM. Advantage grows with K. Confirmed with statistics and negative controls. | Done |
| 03 | Zero-shot generalization | Train K=4,8 → evaluate K=16,32 without retraining. Level 3 check. | Planned |

---

## Project Structure

```
ai-research/
├── README.md
├── docs/                         # Detailed documentation
│   ├── sprint01.md               # Sprint 01 findings
│   ├── sprint02.md               # Sprint 02 breakthrough (full)
│   ├── tests.md                  # All tests and validation
│   └── next_steps.md             # Sprint 03 roadmap
│
├── research_state/               # Active research state
│   ├── BREAKTHROUGH_sprint02.md  # Primary result record
│   ├── sprint02_status.md        # Sprint tracking
│   └── RESUME_PROMPT.md          # Autonomous research prompt
│
├── sprint01/                     # Sprint 01: feasibility study
│   ├── models/                   # AdaptiveModel, MemoryBank, BaseNet
│   ├── tasks/                    # AssociativeRecall, CopyTask
│   ├── experiments/              # Baseline, static, adaptive, difficulty runs
│   ├── training/                 # Trainer, logger, metrics
│   ├── analysis/                 # Statistics, visualizations
│   ├── tests/                    # Unit, gate, stability, baseline tests
│   ├── config/default.yaml
│   ├── results/
│   └── requirements.txt
│
└── sprint02/                     # Sprint 02: breakthrough
    ├── models/
    │   ├── memory_bank_v2.py     # Novelty trigger (max_cos threshold)
    │   ├── adaptive_model_v2.py  # CurriculumRecall model
    │   └── adaptive_model_v3.py  # VariableRecall model (breakthrough)
    ├── tasks/
    │   ├── variable_recall.py    # Fresh dict per batch
    │   └── curriculum_recall.py  # Fixed concept dict
    ├── experiments/
    │   ├── capacity_test.py      # Main breakthrough experiment
    │   ├── stats_analysis.py     # Per-seed Wilcoxon + Cohen's d
    │   ├── negative_controls.py  # Paranoia pass (random inputs, no-write)
    │   ├── slot_ablation.py      # Per-slot accuracy drop
    │   └── fast_adaptation.py    # Zero-shot binding test
    ├── training/trainer_v2.py
    ├── tests/test_curriculum.py  # 19 unit tests
    ├── config/sprint02.yaml
    └── results/
        ├── capacity_test.json    # Main results
        └── stats_analysis.json   # Per-seed data
```

---

## Setup

```bash
pip install -r sprint01/requirements.txt
```

Dependencies: `torch>=2.0.0`, `numpy>=1.24.0`, `scipy>=1.10.0`, `pyyaml>=6.0`, `pytest>=7.0.0`

---

## Reproducing the Breakthrough

```bash
# Main result — AdaptiveV3 vs FixedLSTM on VariableRecall
cd ai-research/sprint02
python experiments/capacity_test.py --K 4 8 16 32 --hd 32 128 --seeds 5 --steps 5000

# Per-seed statistics (Wilcoxon, Cohen's d)
python experiments/stats_analysis.py --K 4 8 16 --hd 32 128 --seeds 5 --steps 5000

# Negative controls (sanity checks)
python experiments/negative_controls.py --K 8 16 --seeds 3 --steps 2000

# Unit tests
pytest tests/
```

All experiments print step-level progress every 500 steps.

---

## Documentation

| Document | Contents |
|----------|----------|
| [docs/sprint01.md](docs/sprint01.md) | Sprint 01 design, results, and why memory was ignored |
| [docs/sprint02.md](docs/sprint02.md) | Sprint 02 breakthrough: full results, stats, architecture |
| [docs/tests.md](docs/tests.md) | All tests run, methodology, pass/fail status |
| [docs/next_steps.md](docs/next_steps.md) | Sprint 03 plan, generalization experiments, open questions |
| [research_state/BREAKTHROUGH_sprint02.md](research_state/BREAKTHROUGH_sprint02.md) | Primary result record with raw data |
