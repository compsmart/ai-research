"""
Slot Specialization Analysis
------------------------------
After training, inspect whether memory slots have specialized to specific
concepts. For the curriculum recall task this is testable: we know the
K fixed concepts and can check whether each slot's key aligns with a
concept key projection.

Metrics:
  1. Concept alignment score: for each active slot, find the concept whose
     key projection best matches the slot's stored key. Report the max cosine
     similarity. If slots are specialized, this should be close to 1.0.

  2. Slot-concept assignment: build a bipartite matching between slots and
     concepts. Each slot is "assigned" to its best-matching concept.
     If K slots each align to a distinct concept, specialization is confirmed.

  3. Value recall accuracy: for each concept, query the memory bank directly
     (bypassing the LSTM) and check whether it returns the correct value.
     This directly tests whether the memory is a usable lookup table.

Usage:
    python analysis/slot_specialization.py --exp curriculum_adaptive_K8 --seed 0
    (run from sprint02/)
"""

import argparse, os, sys, json
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                 '..', 'sprint01'))


def concept_alignment(memory_bank, concept_key_tensors: torch.Tensor) -> dict:
    """
    Args:
        memory_bank: MemoryBankV2 instance
        concept_key_tensors: [K, key_dim] one-hot concept keys
    Returns:
        dict with alignment scores and slot-concept assignments
    """
    if memory_bank.active_count == 0:
        return {"error": "no active slots"}

    active_idx   = memory_bank.active_mask.nonzero(as_tuple=True)[0]
    stored_keys  = memory_bank.slots_key[active_idx]           # [n_active, d_key]
    stored_vals  = memory_bank.slots_value[active_idx]         # [n_active, d_val]

    # Project concept one-hot keys through the learned key_proj
    with torch.no_grad():
        concept_queries = memory_bank.key_proj(concept_key_tensors.float())  # [K, d_key]
        concept_queries = F.normalize(concept_queries, dim=-1)

    slot_keys_norm = F.normalize(stored_keys, dim=-1)           # [n_active, d_key]

    # [K, n_active] cosine similarities
    sims = torch.mm(concept_queries, slot_keys_norm.t())

    # Per-concept best slot and score
    best_slot_per_concept = sims.argmax(dim=-1).tolist()        # [K]
    best_sim_per_concept  = sims.max(dim=-1).values.tolist()    # [K]

    # Per-slot best concept
    best_concept_per_slot = sims.argmax(dim=0).tolist()         # [n_active]
    best_sim_per_slot     = sims.max(dim=0).values.tolist()     # [n_active]

    # Distinctness: are slots assigned to distinct concepts?
    assigned_concepts = set(best_slot_per_concept)
    n_distinct = len(assigned_concepts)

    return {
        "n_active_slots":         memory_bank.active_count,
        "n_concepts":             len(concept_key_tensors),
        "mean_concept_alignment": float(torch.tensor(best_sim_per_concept).mean()),
        "min_concept_alignment":  float(torch.tensor(best_sim_per_concept).min()),
        "n_distinct_assignments": n_distinct,     # <= min(K, n_active)
        "per_concept_best_slot":  best_slot_per_concept,
        "per_concept_best_sim":   [round(x, 4) for x in best_sim_per_concept],
        "per_slot_best_concept":  best_concept_per_slot,
        "per_slot_best_sim":      [round(x, 4) for x in best_sim_per_slot],
    }


def value_recall_accuracy(memory_bank, task) -> dict:
    """
    Query the memory bank directly for each concept (bypassing LSTM).
    Measures whether the bank alone can answer the task.

    Uses a dummy hidden state of zeros + learned key_proj to generate the query.
    A better test projects the actual concept key.
    """
    concept_keys, concept_vals = task.get_concept_tensors()  # [K, key_dim], [K, val_dim]

    correct = 0
    with torch.no_grad():
        for i in range(task.K):
            # Simulate a hidden state that encodes only the query key
            # We use a zero hidden and rely on the input being the one-hot key
            # This is an approximation; full evaluation requires the LSTM.
            query = memory_bank.key_proj(concept_keys[i].unsqueeze(0).float())  # [1, d_key]
            sims  = F.cosine_similarity(
                F.normalize(query, dim=-1),
                F.normalize(memory_bank.slots_key[memory_bank.active_mask], dim=-1),
            ).unsqueeze(0)
            if sims.numel() == 0:
                continue
            attn = torch.softmax(sims / memory_bank.temp, dim=-1)
            ctx  = torch.mm(attn, memory_bank.slots_value[memory_bank.active_mask])
            predicted = ctx.argmax(dim=-1).item()
            expected  = concept_vals[i].argmax().item()
            if predicted == expected:
                correct += 1

    return {
        "direct_memory_accuracy": correct / task.K,
        "correct":                correct,
        "total":                  task.K,
    }


def print_specialization_report(alignment: dict, recall: dict, task) -> None:
    print("\n=== Slot Specialization Report ===")
    print(f"Active slots:        {alignment.get('n_active_slots', 0)}")
    print(f"Concepts (K):        {alignment.get('n_concepts', 0)}")
    print(f"Distinct assignments:{alignment.get('n_distinct_assignments', 0)}")
    print(f"Mean concept align:  {alignment.get('mean_concept_alignment', 0):.4f}")
    print(f"Min concept align:   {alignment.get('min_concept_alignment', 0):.4f}")

    print("\nPer-concept best slot similarity:")
    for i, (slot, sim) in enumerate(zip(
            alignment.get("per_concept_best_slot", []),
            alignment.get("per_concept_best_sim", []))):
        concept_k = task.concept_keys[i]
        concept_v = task.concept_vals[i]
        print(f"  Concept {i} (k={concept_k}->v={concept_v}): "
              f"slot={slot}  sim={sim:.4f}")

    print(f"\nDirect memory recall accuracy: "
          f"{recall.get('correct',0)}/{recall.get('total',0)} = "
          f"{recall.get('direct_memory_accuracy',0):.4f}")

    # Verdict
    n_slots  = alignment.get("n_active_slots", 0)
    n_dist   = alignment.get("n_distinct_assignments", 0)
    mean_sim = alignment.get("mean_concept_alignment", 0.0)
    K        = alignment.get("n_concepts", 0)

    print("\nVerdict:")
    if n_slots >= K and n_dist == K and mean_sim > 0.8:
        print("  SPECIALIZED: Slots align 1-to-1 with concepts. Memory is interpretable.")
    elif n_dist > K // 2 and mean_sim > 0.5:
        print("  PARTIAL: Some slot specialization. Memory partially useful.")
    else:
        print("  NOT SPECIALIZED: Slots do not align with concepts. Memory is noise.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp",         default="curriculum_adaptive_K8")
    parser.add_argument("--seed",        type=int, default=0)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--K",           type=int, default=8)
    parser.add_argument("--n-shown",     type=int, default=4)
    args = parser.parse_args()

    # Import here to avoid circular deps at module load
    from tasks.curriculum_recall import CurriculumRecall
    from models.adaptive_model_v2 import AdaptiveModelV2

    # Reconstruct task with same seed
    task = CurriculumRecall(K=args.K, n_shown=args.n_shown,
                             vocab_size=32, key_dim=32, val_dim=32,
                             concept_seed=args.seed)

    # Load model checkpoint if available; otherwise just build and report placeholder
    checkpoint_path = os.path.join(
        args.results_dir, args.exp, str(args.seed), "model.pt"
    )
    model = AdaptiveModelV2(input_dim=task.input_dim, output_dim=task.output_dim)
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        print(f"Loaded checkpoint: {checkpoint_path}")
    else:
        print(f"No checkpoint at {checkpoint_path}. "
              "Run analysis on a live model instead (see run_curriculum.py).")
        return

    model.eval()
    concept_keys, _ = task.get_concept_tensors()
    alignment = concept_alignment(model.memory, concept_keys)
    recall    = value_recall_accuracy(model.memory, task)
    print_specialization_report(alignment, recall, task)


if __name__ == "__main__":
    main()
