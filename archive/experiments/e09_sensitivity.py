from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from experiments.e06_vision import accuracy, logits_for, make_data
from weight_encodings.single_pbit import SinglePBit


def layer_curve(model, layer_name, module, x, labels, fp_logits, sample_counts,
                batch_size, seed):
    original = module.weight.detach().clone()
    w = original.cpu().numpy().astype(np.float64)
    group_ids = np.broadcast_to(np.arange(w.shape[0]).reshape(
        (-1,) + (1,) * (w.ndim - 1)), w.shape)
    codec = SinglePBit(eps=1e-6)
    encoded = codec.encode(w, group_ids)
    rng = np.random.default_rng(seed)
    running = None
    snapshots = {}
    wanted = set(sample_counts)
    for t in range(1, max(wanted) + 1):
        sampled = codec.sample(encoded, 1, rng)[0]
        module.weight.data.copy_(torch.from_numpy(sampled).to(
            device=original.device, dtype=original.dtype))
        current = logits_for(model, x, batch_size, torch.device("cpu"))
        running = current if running is None else running + current
        if t in wanted:
            mean_logits = running / t
            loss = float(nn.functional.cross_entropy(mean_logits, labels))
            fp_loss = float(nn.functional.cross_entropy(fp_logits, labels))
            snapshots[t] = {
                "accuracy": accuracy(mean_logits, labels),
                "accuracy_delta": accuracy(mean_logits, labels) - accuracy(fp_logits, labels),
                "loss": loss,
                "loss_delta": loss - fp_loss,
                "logit_rmse": float(torch.mean((mean_logits - fp_logits) ** 2).sqrt()),
            }
    module.weight.data.copy_(original)
    return snapshots, encoded.n_groups * 8, w.size


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["mnist", "fashion"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-count", type=int, default=1000)
    parser.add_argument("--sample-counts", nargs="+", type=int, default=[1, 4, 16, 64])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--data", default="data/torchvision")
    parser.add_argument("--checkpoints", default="data/checkpoints")
    parser.add_argument("--out-dir", default="results/sensitivity")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for dataset in args.datasets:
        _, test, model = make_data(dataset, args.data)
        ckpt = Path(args.checkpoints) / f"{dataset}_seed{args.seed}.pt"
        model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
        model.eval()
        rng = np.random.default_rng(args.seed + 1000)
        indices = rng.choice(len(test), size=min(args.eval_count, len(test)), replace=False)
        loader = DataLoader(Subset(test, indices.tolist()), batch_size=args.batch_size,
                            shuffle=False, num_workers=0)
        xb, yb = zip(*[(a, b) for a, b in loader])
        x, labels = torch.cat(xb), torch.cat(yb)
        fp = logits_for(model, x, args.batch_size, torch.device("cpu"))
        fp_acc = accuracy(fp, labels)
        print(f"{dataset}: FP32 accuracy {fp_acc:.4f}", flush=True)
        for layer_index, (name, module) in enumerate(model.named_modules()):
            if not isinstance(module, (nn.Linear, nn.Conv2d)):
                continue
            curve, metadata, nweights = layer_curve(
                model, name, module, x, labels, fp, args.sample_counts,
                args.batch_size, args.seed + 8000 + layer_index)
            for samples in args.sample_counts:
                row = {"dataset": dataset, "seed": args.seed, "layer": name,
                       "samples": samples, "weight_count": nweights,
                       "range_metadata_bytes": metadata,
                       "whole_model_pbits": nweights,
                       "accuracy": curve[samples]["accuracy"],
                       "accuracy_delta": curve[samples]["accuracy_delta"],
                       "cross_entropy": curve[samples]["loss"],
                       "cross_entropy_delta": curve[samples]["loss_delta"],
                       "logit_rmse": curve[samples]["logit_rmse"]}
                records.append(row)
            key = 64 if 64 in curve else max(curve)
            print(f"  {name}: T={key} acc={curve[key]['accuracy']:.4f} "
                  f"dLoss={curve[key]['loss_delta']:+.4f} "
                  f"logitRMSE={curve[key]['logit_rmse']:.3f}", flush=True)
    path = out_dir / "layer_sensitivity.csv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    # Summarize layer vulnerability at the largest simulated sample budget.
    max_t = max(args.sample_counts)
    selected = [row for row in records if row["samples"] == max_t]
    selected.sort(key=lambda row: row["cross_entropy_delta"], reverse=True)
    lines = ["# Layer sensitivity study", "",
             f"One layer at a time was sampled with a single p-bit per weight and "
             f"per-output-channel local ranges. Each stochastic realization was run "
             f"through the otherwise deterministic frozen model; logits were averaged "
             f"over T samples. Evaluation used {args.eval_count} fixed-seed examples.", "",
             f"Results at T={max_t} (largest measured cross-entropy increase first):", "",
             "| Dataset | Layer | Accuracy | Δ accuracy | Δ cross-entropy | Logit RMSE |",
             "|---|---|---:|---:|---:|---:|"]
    for row in selected:
        lines.append(f"| {row['dataset']} | {row['layer']} | {row['accuracy']:.4f} | "
                     f"{row['accuracy_delta']:+.4f} | {row['cross_entropy_delta']:+.4f} | "
                     f"{row['logit_rmse']:.3f} |")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"saved {path}")


if __name__ == "__main__":
    main()
