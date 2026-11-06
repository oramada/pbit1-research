from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from experiments.e06_vision import (accuracy, encode_model_spw, logits_for, make_data)


def collect_subset(dataset, seed: int, count: int, batch_size: int):
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(dataset), size=min(count, len(dataset)), replace=False)
    loader = DataLoader(Subset(dataset, indices.tolist()), batch_size=batch_size,
                        shuffle=False, num_workers=0)
    pairs = list(loader)
    return torch.cat([x for x, _ in pairs]), torch.cat([y for _, y in pairs])


def adaptive_curve(model, encoded, x, labels, device, batch_size, max_samples,
                   mode, threshold, seed, gamma=1.96):
    """Sequential Monte Carlo with per-example stopping on uncertainty or margin."""
    rng = np.random.default_rng(seed)
    modules = dict(model.named_modules())
    originals = {name: modules[name].weight.detach().clone() for name, _, _ in encoded}
    n_examples = len(x)
    mean = None
    m2 = None
    counts = torch.zeros(n_examples, dtype=torch.int32)
    active = torch.ones(n_examples, dtype=torch.bool)
    for _ in range(max_samples):
        for name, codec, state in encoded:
            sample = codec.sample(state, 1, rng)[0]
            modules[name].weight.data.copy_(torch.from_numpy(sample).to(
                device=device, dtype=modules[name].weight.dtype))
        logits = logits_for(model, x, batch_size, device)
        if mean is None:
            mean = torch.zeros_like(logits)
            m2 = torch.zeros_like(logits)
        idx = torch.nonzero(active, as_tuple=False).flatten()
        if idx.numel() == 0:
            break
        new_counts = counts[idx] + 1
        delta = logits[idx] - mean[idx]
        mean[idx] += delta / new_counts[:, None].float()
        m2[idx] += delta * (logits[idx] - mean[idx])
        counts[idx] = new_counts
        if int(counts.max()) < 2:
            continue
        n = counts.clamp(min=2).float()[:, None]
        stderr = torch.sqrt(torch.clamp(m2 / (n - 1.0) / n, min=0.0))
        if mode == "stderr":
            converged = stderr.max(dim=1).values <= threshold
        elif mode == "margin":
            top = mean.topk(2, dim=1)
            class_se = stderr.gather(1, top.indices)
            combined_se = torch.sqrt((class_se ** 2).sum(dim=1))
            converged = (top.values[:, 0] - top.values[:, 1]) > gamma * combined_se
        else:
            raise ValueError(mode)
        active &= ~converged
    for name, old in originals.items():
        modules[name].weight.data.copy_(old)
    final_logits = mean
    return {"accuracy": accuracy(final_logits, labels),
            "avg_samples": float(counts.float().mean()),
            "median_samples": float(counts.median()),
            "p95_samples": float(torch.quantile(counts.float(), .95)),
            "stopped_fraction": float((counts < max_samples).float().mean()),
            "max_samples": int(counts.max())}


def run_dataset(name, seed, args):
    train, test, model = make_data(name, args.data)
    checkpoint = Path(args.checkpoints) / f"{name}_seed{seed}.pt"
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.eval()
    device = torch.device("cpu")
    x, labels = collect_subset(test, seed + 808, args.eval_count, args.batch_size)
    fp = logits_for(model, x, args.batch_size, device)
    fp_acc = accuracy(fp, labels)
    encoded, metadata, n_weights = encode_model_spw(model)
    rows = []
    rows.append({"dataset": name, "seed": seed, "mode": "FP32", "threshold": "",
                 "max_samples": 0, "accuracy": fp_acc, "accuracy_delta": 0.0,
                 "avg_samples": 0, "median_samples": 0, "p95_samples": 0,
                 "stopped_fraction": 0.0, "pbits_per_weight": 0.0,
                 "metadata_bytes": 0, "logical_transitions_per_input": 0})
    configs = [("margin", args.gamma)] + [("stderr", t) for t in args.tolerances]
    for j, (mode, threshold) in enumerate(configs):
        result = adaptive_curve(model, encoded, x, labels, device, args.batch_size,
                                args.max_samples, mode, threshold, seed + 9000 + j,
                                gamma=args.gamma)
        rows.append({"dataset": name, "seed": seed, "mode": mode,
                     "threshold": threshold, "max_samples": args.max_samples,
                     "accuracy": result["accuracy"],
                     "accuracy_delta": result["accuracy"] - fp_acc,
                     "avg_samples": result["avg_samples"],
                     "median_samples": result["median_samples"],
                     "p95_samples": result["p95_samples"],
                     "stopped_fraction": result["stopped_fraction"],
                     "pbits_per_weight": 1.0, "metadata_bytes": metadata,
                     "logical_transitions_per_input": result["avg_samples"] * n_weights})
        print(name, mode, threshold, result, flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["mnist", "fashion"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--eval-count", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=1.96)
    parser.add_argument("--tolerances", nargs="+", type=float,
                        default=[0.1, 0.05, 0.01, 0.005, 0.001])
    parser.add_argument("--data", default="data/torchvision")
    parser.add_argument("--checkpoints", default="data/checkpoints")
    parser.add_argument("--out", default="results/vision/adaptive.csv")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    rows = []
    for dataset in args.datasets:
        for seed in args.seeds:
            rows.extend(run_dataset(dataset, seed, args))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
