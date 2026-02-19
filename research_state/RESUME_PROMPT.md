# Autonomous Research Resume Prompt
# Last updated: 2026-02-18

## Instructions for resuming the AMM research autonomously

You are continuing autonomous AI research on Adaptive Modular Memory (AMM).
Your goal: find a **significant breakthrough** — a Level 2 or Level 3 result
as defined in the research plan at C:\projects\experiments\ai-research\sprint02\

## FIRST: Read these files to orient yourself
1. C:\Users\brad\.claude\projects\C--projects-experiments-ai-research\memory\MEMORY.md
2. C:\projects\experiments\ai-research\research_state\sprint02_status.md
3. C:\projects\experiments\ai-research\research_state\latest_results.md (if exists)

## THEN: Execute the next pending step from sprint02_status.md

## Decision tree:
- If full diagnostic not yet done → run it (see sprint02_status.md "In Progress")
- If diagnostic done but no results analysis → analyze results, update latest_results.md
- If AMM outperforms baselines at K>=16 → run 5-seed experiment for statistical significance
- If extrapolation not yet tested → run it (train K<=16, eval K=32)
- If breakthrough confirmed → write sprint02/RESULTS.md with full findings

## Work rules:
1. ALWAYS update research_state/sprint02_status.md with what you completed and what's next
2. ALWAYS update memory/MEMORY.md if you discover a new key finding or architecture note
3. Keep files CONCISE — remove stale entries, don't add bloat
4. If an experiment fails, diagnose root cause before retrying
5. If all Sprint 02 work is done → define Sprint 03 next steps and write them to
   research_state/sprint03_plan.md

## Working directory: C:\projects\experiments\ai-research\sprint02\
## Run experiments from: cd /c/projects/experiments/ai-research/sprint02

## Key commands:
# Full diagnostic (3 seeds, all K):
cd /c/projects/experiments/ai-research/sprint02 && python experiments/run_curriculum.py --K 4 8 16 32 --seeds 3

# Check results:
cat /c/projects/experiments/ai-research/sprint02/results/curriculum_summary.json

# Unit tests:
cd /c/projects/experiments/ai-research && python -m pytest sprint02/tests/ -v

# Sample efficiency diagnostic (run in sprint02/):
# python analysis/slot_specialization.py --exp curriculum_adaptive_K8 --seed 0

## STOP when: A Level 2+ breakthrough is confirmed OR you've exhausted all Sprint 02 experiments.
## Report findings clearly: what the breakthrough is, what evidence supports it, what p-values.
