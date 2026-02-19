"""
Fast Concept Adaptation Experiment
------------------------------------
Tests the core theoretical advantage of AMM over FixedLSTM:

  Memory banks can bind new concepts in ONE STEP (one write operation),
  without gradient descent. FixedLSTM cannot do this — it requires many
  gradient steps to learn new concept mappings.

Protocol:
  Phase 1 — Pre-train: Both models trained on K concepts (concept_seed=0)
             until 100% accuracy. Weights converge.

  Phase 2 — Concept shift: A NEW concept dictionary (concept_seed=99) is
             introduced. The models have never seen these concept-value bindings.

  Phase 3 — Binding:
    AMM:        Reset memory bank. Write all K new (key, value) pairs to slots
                via a single forward+write pass (no gradient update).
    FixedLSTM:  No update mechanism (no memory). Tested as-is.

  Phase 4 — Test: Evaluate accuracy on new concepts.
    AMM should retrieve the new values from its fresh slots.
    FixedLSTM should fail (no update mechanism for new dict).

  Phase 5 — Gradient recovery: Both models fine-tuned for N steps.
             FixedLSTM gradually recovers. AMM already solved it.

Key metric: Steps to 90% accuracy after concept shift.
  AMM:       0 steps (one-shot, memory write only)
  FixedLSTM: Many steps (gradient descent required)

If AMM achieves high accuracy at 0 gradient steps, this demonstrates
FAST CONCEPT BINDING — a structural capability unavailable to FixedLSTM.
"""

import argparse, sys, os, json
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                 '..', 'sprint01'))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tasks.curriculum_recall import CurriculumRecall
from models.adaptive_model_v2 import AdaptiveModelV2, FixedLSTM


def set_seed(s: int):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


# ---------------------------------------------------------------------------
# Pre-training
# ---------------------------------------------------------------------------

def pretrain(model, task, max_steps=5000, batch_size=32, lr=5e-4, device="cpu"):
    """Train until convergence on the training concept dictionary."""
    model.train()
    opt  = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    crit = nn.CrossEntropyLoss()
    recent_acc = []

    for step in range(1, max_steps + 1):
        inputs, targets = task.generate_batch(batch_size)
        inputs, targets = inputs.to(device), targets.to(device)
        logits = model(inputs)
        loss = crit(logits, targets.argmax(dim=-1))
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

        # Memory write for adaptive model
        if (hasattr(model, "memory") and hasattr(model, "_last_read")
                and model._last_read is not None):
            rd = model._last_read
            d_val = model.memory.d_val
            ts = targets.float()[:, :d_val]
            model.memory.write(query=rd["query"], target_signal=ts,
                               current_loss=loss.item(), attn=rd["attn"],
                               max_attn=rd.get("max_attn", 0.0),
                               max_cos=rd.get("max_cos", 0.0))
            model.memory.step_update(loss.item())

        with torch.no_grad():
            acc = (logits.argmax(-1) == targets.argmax(-1)).float().mean().item()
        recent_acc.append(acc)
        if len(recent_acc) > 200:
            recent_acc.pop(0)

        # Early stopping: 200-step window above 99%
        if step >= 200 and np.mean(recent_acc) >= 0.99:
            return step, float(np.mean(recent_acc))

    return max_steps, float(np.mean(recent_acc))


# ---------------------------------------------------------------------------
# AMM fast binding (no gradient)
# ---------------------------------------------------------------------------

def amm_fast_bind(model, new_task, batch_size=64, device="cpu"):
    """
    Reset memory bank and write all K new (key, value) pairs in a single
    forward pass. No gradient updates to model weights.
    """
    model.eval()
    # Reset all slots
    model.memory.active_mask[:] = False
    model.memory.usage_ema[:]   = 0.0
    model.memory.slot_age[:]    = 0
    model.memory.step           = 0
    model.memory.reset_event_counts()

    # Full binding pass: present all K pairs (n_shown=K-1 so all are in context)
    # We use memory_only_batch: shows exactly 1 pair (the concept) + queries it
    # Better: use a write-only pass with the concept key-value pairs directly
    K = new_task.K
    concept_keys, concept_vals = new_task.get_concept_tensors()  # [K, 32] each

    with torch.no_grad():
        for i in range(K):
            # Simulate a query for concept i (so key_proj sees the concept key)
            query_tensor = concept_keys[i:i+1].float().to(device)  # [1, 32]
            target_tensor = concept_vals[i:i+1].float().to(device) # [1, 32]

            # Compute projected query (same path as forward())
            query = model.memory.key_proj(query_tensor)  # [1, d_key]

            # Read to get attn (needed by write API)
            read_out = model.memory.read(query_tensor)

            # Write: force novelty (loss=1.0 above floor, max_cos=0.0 → novel)
            model.memory.write(
                query=query,
                target_signal=target_tensor,
                current_loss=1.0,
                attn=read_out["attn"],
                max_attn=0.0,
                max_cos=0.0,    # force novelty: always create new slot
            )

    return model.memory.active_count


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model, task, n_batches=50, batch_size=64, device="cpu"):
    model.eval()
    correct = 0
    total   = 0
    with torch.no_grad():
        for _ in range(n_batches):
            inputs, targets = task.generate_batch(batch_size)
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            correct += (logits.argmax(-1) == targets.argmax(-1)).sum().item()
            total   += targets.size(0)
    return correct / total


def fine_tune_steps_to_90(model, task, max_steps=3000, batch_size=32,
                           lr=5e-4, device="cpu"):
    """Fine-tune on new concept dict; return steps until 90% window accuracy."""
    model.train()
    opt  = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    crit = nn.CrossEntropyLoss()
    recent = []
    for step in range(1, max_steps + 1):
        inputs, targets = task.generate_batch(batch_size)
        inputs, targets = inputs.to(device), targets.to(device)
        logits = model(inputs)
        loss = crit(logits, targets.argmax(dim=-1))
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        with torch.no_grad():
            acc = (logits.argmax(-1) == targets.argmax(-1)).float().mean().item()
        recent.append(acc)
        if len(recent) > 100:
            recent.pop(0)
        if len(recent) >= 100 and np.mean(recent) >= 0.90:
            return step
    return max_steps


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_experiment(K: int, seed: int, train_seed: int = 0,
                   new_seed: int = 99, device: str = "cpu") -> dict:
    set_seed(seed)

    # Training task
    train_task = CurriculumRecall(K=K, n_shown=K//2, vocab_size=32,
                                   key_dim=32, val_dim=32, concept_seed=train_seed)
    # New (shifted) concept dict
    new_task   = CurriculumRecall(K=K, n_shown=K//2, vocab_size=32,
                                   key_dim=32, val_dim=32, concept_seed=new_seed)

    results = {"K": K, "seed": seed}

    # -----------------------------------------------------------------------
    # AdaptiveModelV2
    # -----------------------------------------------------------------------
    set_seed(seed)
    amm = AdaptiveModelV2(input_dim=train_task.input_dim, output_dim=train_task.output_dim,
                           key_dim=32, d_val=32, novelty_threshold=0.5, loss_floor=0.3)

    steps, train_acc = pretrain(amm, train_task, max_steps=5000, device=device)
    results["amm_pretrain_steps"]     = steps
    results["amm_pretrain_acc"]       = round(train_acc, 4)
    results["amm_old_acc_before"]     = round(evaluate(amm, train_task, device=device), 4)
    results["amm_new_acc_before"]     = round(evaluate(amm, new_task,  device=device), 4)

    n_bound = amm_fast_bind(amm, new_task, device=device)
    results["amm_slots_bound"]        = n_bound
    results["amm_new_acc_after_bind"] = round(evaluate(amm, new_task, device=device), 4)

    # -----------------------------------------------------------------------
    # FixedLSTM
    # -----------------------------------------------------------------------
    set_seed(seed)
    lstm = FixedLSTM(input_dim=train_task.input_dim, output_dim=train_task.output_dim)

    steps, train_acc = pretrain(lstm, train_task, max_steps=5000, device=device)
    results["lstm_pretrain_steps"]     = steps
    results["lstm_pretrain_acc"]       = round(train_acc, 4)
    results["lstm_old_acc_before"]     = round(evaluate(lstm, train_task, device=device), 4)
    results["lstm_new_acc_before"]     = round(evaluate(lstm, new_task,  device=device), 4)

    # Fine-tune both on new dict; track steps to 90%
    set_seed(seed)
    amm_finetune = AdaptiveModelV2(input_dim=train_task.input_dim,
                                    output_dim=train_task.output_dim,
                                    key_dim=32, d_val=32,
                                    novelty_threshold=0.5, loss_floor=0.3)
    pretrain(amm_finetune, train_task, max_steps=5000, device=device)
    amm_fast_bind(amm_finetune, new_task, device=device)  # bind first
    results["amm_finetune_to_90"] = fine_tune_steps_to_90(amm_finetune, new_task, device=device)

    results["lstm_finetune_to_90"] = fine_tune_steps_to_90(lstm, new_task, device=device)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--K",      type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--seeds",  type=int, default=3)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    all_results = []
    for K in args.K:
        for seed in range(args.seeds):
            print("K={} seed={}...".format(K, seed), end=" ", flush=True)
            r = run_experiment(K=K, seed=seed, device=args.device)
            all_results.append(r)
            print("AMM bind acc={:.3f}  LSTM new acc={:.3f}  "
                  "AMM ft={} LSTM ft={}".format(
                      r["amm_new_acc_after_bind"], r["lstm_new_acc_before"],
                      r["amm_finetune_to_90"], r["lstm_finetune_to_90"]))

    print("\n=== Fast Adaptation Summary ===")
    print("{:>4}  {:>12}  {:>12}  {:>12}  {:>12}".format(
        "K", "AMM_bind_acc", "LSTM_new_acc", "AMM_ft90", "LSTM_ft90"))
    print("-" * 60)
    for K in args.K:
        k_res = [r for r in all_results if r["K"] == K]
        amm_bind  = np.mean([r["amm_new_acc_after_bind"] for r in k_res])
        lstm_new  = np.mean([r["lstm_new_acc_before"]     for r in k_res])
        amm_ft    = np.mean([r["amm_finetune_to_90"]      for r in k_res])
        lstm_ft   = np.mean([r["lstm_finetune_to_90"]     for r in k_res])
        print("{:>4}  {:>12.4f}  {:>12.4f}  {:>12.0f}  {:>12.0f}".format(
            K, amm_bind, lstm_new, amm_ft, lstm_ft))

    print("\nHypothesis: AMM_bind_acc >> LSTM_new_acc at all K values")
    print("(AMM can bind new concepts in 0 gradient steps; LSTM cannot)")

    out = "results/fast_adaptation.json"
    os.makedirs("results", exist_ok=True)
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nResults -> {}".format(out))


if __name__ == "__main__":
    main()
