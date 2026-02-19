"""
Trigger Ablation
-----------------
Compares the novelty trigger against the sprint01 loss-relative trigger
on the curriculum recall task, to isolate the trigger mechanism's contribution.

Variants:
  novelty     — max_attn < threshold  (sprint02 default)
  loss_rel    — loss > threshold * running_mean  (sprint01 default)
  no_memory   — FixedLSTM (memory completely absent)

Usage:
    python experiments/run_ablation.py --K 8 --seeds 3
"""

import argparse, os, sys, json, random
import numpy as np
import torch
import torch.nn as nn

_sprint02_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_sprint01_dir = os.path.join(_sprint02_dir, '..', 'sprint01')
sys.path.insert(0, _sprint01_dir)
sys.path.insert(0, _sprint02_dir)

import yaml
from tasks.curriculum_recall import CurriculumRecall
from models.adaptive_model_v2 import AdaptiveModelV2, FixedLSTM
from models.memory_bank_v2 import MemoryBankV2
from training.trainer_v2 import TrainerV2


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def load_cfg(path):
    with open(path) as f:
        return yaml.safe_load(f)


class AdaptiveModelLossRelative(nn.Module):
    """
    AdaptiveModelV2 wired with the sprint01 loss-relative trigger.
    Uses MemoryBankV2 but overrides write() to use the relative threshold.
    """
    def __init__(self, input_dim, output_dim, hidden_dim=128,
                 max_slots=100, d_key=32, d_val=32, temp=0.1,
                 error_threshold=0.5, **kwargs):
        super().__init__()
        self.encoder = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim,
                               num_layers=1, batch_first=True)
        # Use MemoryBankV2 but we'll call the sprint01-style write externally
        self.memory = MemoryBankV2(
            max_slots=max_slots, d_key=d_key, d_val=d_val, temp=temp,
            novelty_threshold=0.0,   # effectively disabled
            loss_floor=0.0,          # effectively disabled
            hidden_dim=hidden_dim, **kwargs
        )
        self.error_threshold = error_threshold
        self.output_head = nn.Linear(hidden_dim + d_val, output_dim)
        self._last_read = None

    def forward(self, inputs):
        _, (h_n, _) = self.encoder(inputs)
        final_h = h_n[-1]
        read_out = self.memory.read(final_h)
        self._last_read = read_out
        ctx = read_out["ctx"]
        return self.output_head(torch.cat([final_h, ctx], dim=-1))

    def write_loss_relative(self, query, target_signal, current_loss, attn):
        """Sprint01-style: grow when loss > threshold * running_mean."""
        n_active   = self.memory.active_count
        high_error = current_loss > self.error_threshold * float(self.memory._running_mean)
        can_grow   = n_active < self.memory.max_slots

        import torch.nn.functional as F
        with torch.no_grad():
            if high_error and can_grow:
                inactive = (~self.memory.active_mask).nonzero(as_tuple=True)[0]
                if len(inactive) > 0:
                    s = inactive[0].item()
                    self.memory.slots_key[s]   = F.normalize(query.detach().mean(0), dim=-1)
                    self.memory.slots_value[s] = target_signal.detach().mean(0)
                    self.memory.active_mask[s] = True
                    self.memory.usage_ema[s]   = 0.0
                    self.memory.slot_age[s]    = 0
                    self.memory.write_events  += 1
            else:
                self.memory._soft_update_best(attn, target_signal)
        self.memory._maybe_age_slots()


def run_one(variant: str, K: int, seed: int, cfg: dict,
            results_dir: str, device: str) -> dict:
    set_seed(seed)
    cr = cfg["curriculum_recall"]
    mem = cfg["memory"]
    hidden = cfg["model"]["hidden_dim"]

    task = CurriculumRecall(
        K=K, n_shown=K // 2,
        vocab_size=cr["vocab_size"], key_dim=cr["key_dim"], val_dim=cr["val_dim"],
        concept_seed=seed, device=device,
    )

    if variant == "no_memory":
        model = FixedLSTM(input_dim=task.input_dim, output_dim=task.output_dim,
                          hidden_dim=hidden)
    elif variant == "novelty":
        model = AdaptiveModelV2(
            input_dim=task.input_dim, output_dim=task.output_dim,
            hidden_dim=hidden, max_slots=mem["max_slots"],
            d_key=mem["d_key"], d_val=mem["d_val"], temp=mem["temp"],
            novelty_threshold=mem["novelty_threshold"], loss_floor=mem["loss_floor"],
            usage_ema_decay=mem["usage_ema_decay"], min_usage=mem["min_usage"],
            min_age=mem["min_age"], prune_every=mem["prune_every"],
            merge_threshold=mem["merge_threshold"], merge_every=mem["merge_every"],
            write_lr=mem["write_lr"],
        )
    elif variant == "loss_rel":
        model = AdaptiveModelLossRelative(
            input_dim=task.input_dim, output_dim=task.output_dim,
            hidden_dim=hidden, max_slots=mem["max_slots"],
            d_key=mem["d_key"], d_val=mem["d_val"], temp=mem["temp"],
            error_threshold=0.5,
            usage_ema_decay=mem["usage_ema_decay"], min_usage=mem["min_usage"],
            min_age=mem["min_age"], prune_every=mem["prune_every"],
            merge_threshold=mem["merge_threshold"], merge_every=mem["merge_every"],
            write_lr=mem["write_lr"],
        )
        # Inject running_mean buffer so write_loss_relative works
        import torch
        model.memory.register_buffer("_running_mean", torch.tensor(1.0))
    else:
        raise ValueError(f"Unknown variant: {variant}")

    trainer = TrainerV2(
        model=model, task=task, cfg=cfg,
        experiment_name=f"ablation_{variant}_K{K}",
        seed=seed, results_dir=results_dir, task_K=K, device=device,
    )
    results = trainer.train()
    results.update({"variant": variant, "K": K, "seed": seed})
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--K",           type=int, nargs="+", default=[8])
    parser.add_argument("--seeds",       type=int, default=3)
    parser.add_argument("--variants",    nargs="+",
                        default=["no_memory", "loss_rel", "novelty"])
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--config",      default="config/sprint02.yaml")
    parser.add_argument("--device",      default="cpu")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    os.makedirs(args.results_dir, exist_ok=True)
    all_results = {}

    for variant in args.variants:
        all_results[variant] = {}
        for K in args.K:
            seed_results = []
            for seed in range(args.seeds):
                print(f"\n--- ablation:{variant} | K={K} | seed={seed} ---")
                r = run_one(variant, K, seed, cfg, args.results_dir, args.device)
                seed_results.append(r)
                print(f"    acc={r.get('final_acc',0):.4f}  "
                      f"slots={r.get('slot_count_max',0)}")
            accs = [r.get("final_acc", 0) for r in seed_results]
            all_results[variant][K] = {
                "acc_mean": float(np.mean(accs)),
                "acc_std":  float(np.std(accs)),
                "slot_mean": float(np.mean([r.get("slot_count_max",0) for r in seed_results])),
            }

    print("\n=== Trigger Ablation Results ===")
    for K in args.K:
        print(f"\nK={K}:")
        for v in args.variants:
            s = all_results[v].get(K, {})
            print(f"  {v:12}: acc={s.get('acc_mean',0):.4f} +/- {s.get('acc_std',0):.4f}"
                  f"  slots={s.get('slot_mean',0):.1f}")

    out_path = os.path.join(args.results_dir, "ablation_summary.json")
    with open(out_path, "w") as f:
        json.dump({v: {str(k): s for k,s in by_k.items()}
                   for v, by_k in all_results.items()}, f, indent=2)
    print(f"\nSaved -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
