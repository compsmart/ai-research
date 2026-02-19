"""
Paranoia pass — negative controls for the VariableRecall breakthrough.

Control 1: Shuffled labels — both models should go to ~chance (3.125% for 32 classes)
Control 2: Random inputs  — both models should go to ~chance
Control 3: AMM with writes disabled (memory always empty) — should degrade to LSTM-level

If Control 1 or 2 yield non-chance accuracy: eval bug (wrong axis, label leak, cached target).
If Control 3 does NOT degrade: LSTM is doing all the work, memory claim is false.
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

from tasks.variable_recall import VariableRecall
from models.adaptive_model_v3 import AdaptiveModelV3, FixedLSTM


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def run_control(K, model, task, max_steps=2000, batch_size=64,
                shuffle_labels=False, random_inputs=False, disable_writes=False):
    opt  = optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    recent = []

    for step in range(1, max_steps + 1):
        model.train()
        x, y = task.generate_batch(batch_size)

        if random_inputs:
            x = torch.randn_like(x)
        if shuffle_labels:
            idx = torch.randperm(y.size(0))
            y   = y[idx]
        if disable_writes and hasattr(model, '_write_support_pairs'):
            # Monkey-patch to skip writing
            orig = model._write_support_pairs
            model._write_support_pairs = lambda inp: None

        logits = model(x)

        if disable_writes and hasattr(model, '_write_support_pairs'):
            model._write_support_pairs = orig

        loss = crit(logits, y.argmax(-1))
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

        with torch.no_grad():
            acc = (logits.argmax(-1) == y.argmax(-1)).float().mean().item()
        recent.append(acc)
        if len(recent) > 200:
            recent.pop(0)

    return float(np.mean(recent[-200:]))


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--K",     nargs="+", type=int, default=[8])
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--steps", type=int, default=2000)
    args = parser.parse_args()

    chance = 1.0 / 32  # vocab_size=32

    print(f"Chance level: {chance:.4f} ({chance*100:.2f}%)", flush=True)
    print(f"K={args.K}  seeds={args.seeds}  steps={args.steps}", flush=True)
    print("=" * 70, flush=True)

    controls = [
        ("shuffled_labels", dict(shuffle_labels=True)),
        ("random_inputs",   dict(random_inputs=True)),
        ("amm_no_write",    dict(disable_writes=True)),
    ]

    for K in args.K:
        task = VariableRecall(K=K, vocab_size=32, key_dim=32, val_dim=32)
        print(f"\nK={K}", flush=True)

        for ctrl_name, ctrl_kwargs in controls:
            accs_lstm = []
            accs_amm  = []
            for s in range(args.seeds):
                set_seed(s)
                lstm = FixedLSTM(task.input_dim, task.output_dim, hidden_dim=64)
                amm  = AdaptiveModelV3(task.input_dim, task.output_dim,
                                       hidden_dim=64, key_dim=32)
                acc_l = run_control(K, lstm, task, max_steps=args.steps,
                                    **ctrl_kwargs)
                set_seed(s)
                acc_a = run_control(K, amm,  task, max_steps=args.steps,
                                    **ctrl_kwargs)
                accs_lstm.append(acc_l)
                accs_amm.append(acc_a)

            mean_l = np.mean(accs_lstm)
            mean_a = np.mean(accs_amm)
            pass_l = abs(mean_l - chance) < 0.05
            pass_a = abs(mean_a - chance) < 0.05

            if ctrl_name == "amm_no_write":
                # Pass = AMM degrades significantly (memory was doing work)
                normal_amm = 0.99  # expected from capacity test
                degraded   = mean_a < normal_amm - 0.20
                verdict = "MEMORY CONTRIBUTES" if degraded else "LSTM DOING ALL WORK"
                print(f"  {ctrl_name:20s}: AMM={mean_a:.4f}  LSTM={mean_l:.4f}  "
                      f"-> {verdict}", flush=True)
            else:
                # Pass = both near chance
                status_l = "PASS" if pass_l else "FAIL(leak?)"
                status_a = "PASS" if pass_a else "FAIL(leak?)"
                print(f"  {ctrl_name:20s}: LSTM={mean_l:.4f}({status_l})  "
                      f"AMM={mean_a:.4f}({status_a})", flush=True)

    print("\n--- Interpretation ---", flush=True)
    print("shuffled_labels/random_inputs PASS -> no label leak or eval bug", flush=True)
    print("amm_no_write MEMORY CONTRIBUTES -> memory is the active ingredient", flush=True)


if __name__ == "__main__":
    main()
