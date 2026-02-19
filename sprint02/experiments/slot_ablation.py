"""
Slot Ablation Study
---------------------
After training AMM on K=16 concepts, ablate one slot at a time and
measure which concepts lose accuracy.

Hypothesis (Level 2):
  Slot i encodes EXACTLY one concept. Removing slot i causes accuracy
  drop for that concept only, not others. This demonstrates:
  1. Slots are factorized (one concept per slot, not distributed)
  2. Memory is interpretable (slots align to semantic concepts)
  3. Each slot encodes an independent, removable unit of knowledge

Comparison:
  - AMM: ablating slot i → only concept i loses accuracy
  - FixedLSTM: no slots to ablate, but accuracy drop from
    "removing the representation of concept i from hidden state"
    is distributed across all concepts

Protocol:
  1. Train AMM until convergence (K=16, concept_seed=0)
  2. Identify which slot aligns to each concept (cosine sim to key_proj output)
  3. Ablate each slot: set active_mask[i] = False
  4. Measure per-concept accuracy after ablation
  5. Compute specificity = accuracy[concept_i] drop vs other concepts' drop

Strong result: specificity > 0.8 (ablating slot i primarily hurts concept i)
Weak result: specificity ~0.5 (distributed, non-interpretable)
"""

import sys, os, json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                 '..', 'sprint01'))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tasks.curriculum_recall import CurriculumRecall
from models.adaptive_model_v2 import AdaptiveModelV2


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_to_convergence(model, task, max_steps=8000, batch_size=32, lr=5e-4,
                          patience_window=500, patience_threshold=0.99, device="cpu"):
    opt  = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    crit = nn.CrossEntropyLoss()
    recent_acc = []

    for step in range(1, max_steps + 1):
        model.train()
        inputs, targets = task.generate_batch(batch_size)
        inputs, targets = inputs.to(device), targets.to(device)
        logits = model(inputs)
        loss = crit(logits, targets.argmax(-1))
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

        if (hasattr(model, "memory") and hasattr(model, "_last_read")
                and model._last_read is not None):
            rd   = model._last_read
            d_val = model.memory.d_val
            ts   = targets.float()[:, :d_val]
            model.memory.write(rd["query"], ts, loss.item(), rd["attn"],
                               max_attn=rd.get("max_attn", 0.0),
                               max_cos=rd.get("max_cos", 0.0))
            model.memory.step_update(loss.item())

        with torch.no_grad():
            acc = (logits.argmax(-1) == targets.argmax(-1)).float().mean().item()
        recent_acc.append(acc)
        if len(recent_acc) > patience_window:
            recent_acc.pop(0)
        if len(recent_acc) >= patience_window and np.mean(recent_acc) >= patience_threshold:
            return step, float(np.mean(recent_acc))

    return max_steps, float(np.mean(recent_acc)) if recent_acc else 0.0


# ---------------------------------------------------------------------------
# Per-concept accuracy
# ---------------------------------------------------------------------------

def per_concept_accuracy(model, task, n_trials=200, device="cpu"):
    """
    For each concept k, generate n_trials episodes where k is the query,
    and measure accuracy. Returns dict: concept_key -> accuracy.
    """
    model.eval()
    concept_keys = list(task.concept_dict.keys())
    accuracies = {}

    with torch.no_grad():
        for ck in concept_keys:
            correct = 0
            for _ in range(n_trials):
                # Generate a batch of 1 where the query IS concept ck
                # We do this by calling generate_batch and filtering, but that's
                # slow. Instead, construct the episode directly.
                inputs, targets = _make_episode_for_concept(task, ck)
                inputs  = inputs.to(device)
                targets = targets.to(device)
                logits  = model(inputs)
                pred    = logits.argmax(-1).item()
                expected = targets.argmax(-1).item()
                if pred == expected:
                    correct += 1
            accuracies[ck] = correct / n_trials

    return accuracies


def _make_episode_for_concept(task, query_concept_key: int):
    """
    Build a single-sample episode where the query is `query_concept_key`.
    n_shown pairs are sampled from OTHER concepts (to force memory retrieval
    when query_concept_key is NOT among the shown pairs).
    """
    import random as rng
    K = task.K
    all_keys = list(task.concept_dict.keys())
    other_keys = [k for k in all_keys if k != query_concept_key]

    # Show n_shown pairs from other concepts
    n_shown = task.n_shown
    if n_shown >= len(other_keys):
        shown_keys = other_keys[:n_shown]
    else:
        shown_keys = rng.sample(other_keys, n_shown)

    batch_size = 1
    seq_len    = n_shown + 1
    key_dim    = task.key_dim
    val_dim    = task.val_dim
    inputs     = torch.zeros(batch_size, seq_len, key_dim + val_dim)
    targets    = torch.zeros(batch_size, val_dim)

    for t, sk in enumerate(shown_keys):
        sv = task.concept_dict[sk]
        inputs[0, t, sk]              = 1.0
        inputs[0, t, key_dim + sv]    = 1.0

    # Last step: query (key only, no value)
    inputs[0, -1, query_concept_key] = 1.0
    expected_val = task.concept_dict[query_concept_key]
    targets[0, expected_val] = 1.0

    return inputs, targets


# ---------------------------------------------------------------------------
# Slot alignment
# ---------------------------------------------------------------------------

def compute_slot_concept_alignment(model, task) -> dict:
    """
    For each active slot, find the concept it best aligns with.
    Returns: {slot_global_idx: (concept_key, cosine_sim)}
    """
    concept_keys_list = list(task.concept_dict.keys())
    # [K, key_dim] one-hot concept keys
    K        = len(concept_keys_list)
    key_dim  = task.key_dim
    ch       = torch.zeros(K, key_dim)
    for i, ck in enumerate(concept_keys_list):
        ch[i, ck] = 1.0

    alignment = {}
    with torch.no_grad():
        concept_queries = model.memory.key_proj(ch)                    # [K, d_key]
        concept_queries = F.normalize(concept_queries, dim=-1)

        active_idx = model.memory.active_mask.nonzero(as_tuple=True)[0]
        slot_keys  = F.normalize(model.memory.slots_key[active_idx], dim=-1)  # [n, d_key]

        sims = torch.mm(concept_queries, slot_keys.t())  # [K, n]
        for local_slot, global_slot in enumerate(active_idx.tolist()):
            best_concept_local = sims[:, local_slot].argmax().item()
            best_concept_key   = concept_keys_list[best_concept_local]
            best_sim           = sims[best_concept_local, local_slot].item()
            alignment[global_slot] = {
                "concept_key": best_concept_key,
                "concept_val": task.concept_dict[best_concept_key],
                "cosine_sim":  round(best_sim, 4),
            }
    return alignment


# ---------------------------------------------------------------------------
# Ablation
# ---------------------------------------------------------------------------

def run_slot_ablation(model, task, alignment: dict,
                      n_trials_per_concept: int = 100, device: str = "cpu") -> dict:
    """
    For each active slot:
      1. Record baseline per-concept accuracy (full model)
      2. Ablate slot (set active_mask = False)
      3. Measure post-ablation per-concept accuracy
      4. Restore slot
      5. Compute specificity: drop for aligned concept vs drop for others
    """
    # Baseline
    baseline = per_concept_accuracy(model, task, n_trials=n_trials_per_concept, device=device)

    ablation_results = {}
    active_idx = model.memory.active_mask.nonzero(as_tuple=True)[0].tolist()

    for slot_i in active_idx:
        info = alignment.get(slot_i, {})
        aligned_concept = info.get("concept_key")

        # Ablate
        model.memory.active_mask[slot_i] = False
        post = per_concept_accuracy(model, task, n_trials=n_trials_per_concept, device=device)
        # Restore
        model.memory.active_mask[slot_i] = True

        # Compute drops
        drops = {}
        for ck in baseline:
            drops[ck] = baseline[ck] - post.get(ck, 0.0)

        aligned_drop = drops.get(aligned_concept, 0.0)
        other_drops  = [d for ck, d in drops.items() if ck != aligned_concept]
        mean_other   = float(np.mean(other_drops)) if other_drops else 0.0
        specificity  = aligned_drop / (aligned_drop + abs(mean_other) + 1e-8)

        ablation_results[slot_i] = {
            "aligned_concept": aligned_concept,
            "cosine_sim": info.get("cosine_sim", 0.0),
            "baseline_acc_aligned":  round(baseline.get(aligned_concept, 0.0), 4),
            "post_ablation_acc":     round(post.get(aligned_concept, 0.0) if aligned_concept else 0.0, 4),
            "aligned_drop":          round(aligned_drop, 4),
            "mean_other_drop":       round(mean_other, 4),
            "specificity":           round(specificity, 4),
        }

    return ablation_results, baseline


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--K",     type=int, default=16)
    parser.add_argument("--seed",  type=int, default=0)
    parser.add_argument("--steps", type=int, default=8000)
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cpu"

    task = CurriculumRecall(K=args.K, n_shown=args.K // 2, vocab_size=32,
                             key_dim=32, val_dim=32, concept_seed=args.seed)

    model = AdaptiveModelV2(
        input_dim=task.input_dim, output_dim=task.output_dim,
        key_dim=32, d_val=32, novelty_threshold=0.5, loss_floor=0.3,
    )

    print("Training AMM on K={} concepts (seed={})...".format(args.K, args.seed))
    steps, acc = train_to_convergence(model, task, max_steps=args.steps)
    print("Converged: steps={}, acc={:.4f}, active_slots={}".format(
        steps, acc, model.memory.active_count))

    # Slot alignment
    alignment = compute_slot_concept_alignment(model, task)
    print("\nSlot-Concept Alignment:")
    for slot_i, info in sorted(alignment.items()):
        print("  slot {:3d} -> concept {:2d} (val={:2d})  sim={:.4f}".format(
            slot_i, info["concept_key"], info["concept_val"], info["cosine_sim"]))

    # Ablation
    print("\nRunning per-slot ablation study...")
    ablation, baseline = run_slot_ablation(model, task, alignment, n_trials_per_concept=100)

    print("\nBaseline per-concept accuracy:")
    for ck in sorted(baseline):
        print("  concept {:2d}: {:.3f}".format(ck, baseline[ck]))

    print("\nAblation specificity (1.0 = perfectly specific to one concept):")
    specificities = []
    for slot_i in sorted(ablation):
        r = ablation[slot_i]
        specificities.append(r["specificity"])
        print("  slot {:3d} (concept {:2d}, sim={:.3f}): "
              "drop={:.3f} (aligned) vs {:.3f} (others)  "
              "specificity={:.3f}".format(
                  slot_i, r["aligned_concept"] if r["aligned_concept"] is not None else -1,
                  r["cosine_sim"],
                  r["aligned_drop"], r["mean_other_drop"],
                  r["specificity"]))

    mean_spec = np.mean(specificities) if specificities else 0.0
    print("\nMean specificity across all slots: {:.4f}".format(mean_spec))

    if mean_spec >= 0.8:
        verdict = "FACTORIZED: Slots encode individual concepts. Memory is interpretable."
    elif mean_spec >= 0.5:
        verdict = "PARTIAL: Some slot specialization. Memory partially interpretable."
    else:
        verdict = "DISTRIBUTED: Slots do not specialize. Memory not interpretable."
    print("Verdict:", verdict)

    # Save
    os.makedirs("results", exist_ok=True)
    out = {
        "K": args.K, "seed": args.seed, "steps": steps, "final_acc": acc,
        "active_slots": model.memory.active_count,
        "alignment": {str(k): v for k, v in alignment.items()},
        "ablation": {str(k): v for k, v in ablation.items()},
        "baseline_accuracy": {str(k): v for k, v in baseline.items()},
        "mean_specificity": float(mean_spec),
        "verdict": verdict,
    }
    path = "results/slot_ablation_K{}_seed{}.json".format(args.K, args.seed)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print("Results ->", path)


if __name__ == "__main__":
    main()
