"""
Capacity Bottleneck Test: Variable-Dict Recall
-----------------------------------------------
Tests whether AdaptiveModelV3 outperforms FixedLSTM on in-context
associative recall when the concept dictionary changes every batch.

Key prediction:
  - FixedLSTM: accuracy degrades as K increases (must compress K pairs into hidden_dim)
  - AdaptiveV3: accuracy maintained as K increases (explicit slots, no compression)

This is the genuine structural advantage of the memory bank.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                 '..', 'sprint01'))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import json

from tasks.variable_recall import VariableRecall
from models.adaptive_model_v3 import AdaptiveModelV3, FixedLSTM


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def run(K, hidden_dim, model_cls, max_steps=5000, batch_size=64, lr=1e-3, seed=0,
        log_every=500, label=None):
    set_seed(seed)
    task = VariableRecall(K=K, vocab_size=32, key_dim=32, val_dim=32)
    if model_cls == FixedLSTM:
        model = FixedLSTM(task.input_dim, task.output_dim, hidden_dim=hidden_dim)
    else:
        model = AdaptiveModelV3(task.input_dim, task.output_dim,
                                 hidden_dim=hidden_dim, key_dim=32)
    opt  = optim.Adam(model.parameters(), lr=lr, weight_decay=0)
    crit = nn.CrossEntropyLoss()
    recent = []
    prefix = label or "{}".format(model_cls.__name__)
    for step in range(1, max_steps + 1):
        model.train()
        x, y = task.generate_batch(batch_size)
        logits = model(x)
        loss   = crit(logits, y.argmax(-1))
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        with torch.no_grad():
            acc = (logits.argmax(-1) == y.argmax(-1)).float().mean().item()
        recent.append(acc)
        if len(recent) > 200:
            recent.pop(0)
        if log_every and step % log_every == 0:
            print("  [{}/{}] {} loss={:.4f} acc={:.3f}".format(
                step, max_steps, prefix, loss.item(), acc), flush=True)
    return float(np.mean(recent[-200:]))


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--K",      nargs="+", type=int, default=[4, 8, 16])
    parser.add_argument("--hd",     nargs="+", type=int, default=[32, 128])
    parser.add_argument("--seeds",  type=int, default=3)
    parser.add_argument("--steps",  type=int, default=5000)
    args = parser.parse_args()

    results = {}
    print("K  hd   FixedLSTM  AdaptiveV3   delta", flush=True)
    print("-" * 45, flush=True)

    for K in args.K:
        for hd in args.hd:
            acc_lstm = np.mean([run(K, hd, FixedLSTM, max_steps=args.steps, seed=s)
                                for s in range(args.seeds)])
            acc_amm  = np.mean([run(K, hd, AdaptiveModelV3, max_steps=args.steps, seed=s)
                                for s in range(args.seeds)])
            delta = acc_amm - acc_lstm
            key = "K{}_hd{}".format(K, hd)
            results[key] = {"lstm": round(acc_lstm, 4), "amm": round(acc_amm, 4),
                            "delta": round(delta, 4)}
            print("K={:2d} hd={:3d}  {:.4f}     {:.4f}     {:+.4f}".format(
                K, hd, acc_lstm, acc_amm, delta), flush=True)

    # Summary
    print("\n=== Verdict ===", flush=True)
    all_deltas = [v["delta"] for v in results.values()]
    mean_delta = np.mean(all_deltas)
    print("Mean AMM advantage: {:+.4f}".format(mean_delta), flush=True)
    if mean_delta > 0.10:
        print("BREAKTHROUGH: AMM consistently outperforms FixedLSTM on variable-dict recall.", flush=True)
        print("The explicit slot mechanism provides a genuine capacity advantage.", flush=True)
    elif mean_delta > 0.05:
        print("PARTIAL: AMM has modest advantage. Need more training or larger K.", flush=True)
    else:
        print("NULL: No clear AMM advantage. Architecture may need redesign.", flush=True)

    os.makedirs("results", exist_ok=True)
    out_path = "results/capacity_test.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print("Results saved to", out_path, flush=True)


if __name__ == "__main__":
    main()
