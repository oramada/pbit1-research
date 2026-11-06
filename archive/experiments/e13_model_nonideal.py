from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from experiments.e06_vision import accuracy, encode_model_spw, logits_for, make_data
from stochastic.model_correlated import MarkovPBitState, theoretical_effective_samples


def run_rho(model, encoded, x, labels, n_samples, rho, seed, batch_size=256):
    rng = np.random.default_rng(seed)
    modules = dict(model.named_modules())
    originals = {name: modules[name].weight.detach().clone() for name, _, _ in encoded}
    chains = {name: MarkovPBitState(state.probabilities, rho, rng)
              for name, _, state in encoded}
    running = None
    for _ in range(n_samples):
        for name, codec, state in encoded:
            bits = chains[name].next()
            spins = 2.0 * bits - 1.0
            c = state.centers[state.group_ids]
            r = state.radii[state.group_ids]
            sampled_weight = c + r * spins
            modules[name].weight.data.copy_(torch.from_numpy(sampled_weight).to(
                device=modules[name].weight.device,
                dtype=modules[name].weight.dtype))
        logits = logits_for(model, x, batch_size, torch.device("cpu"))
        running = logits if running is None else running + logits
    for name, old in originals.items():
        modules[name].weight.data.copy_(old)
    mean_logits = running / n_samples
    return mean_logits, accuracy(mean_logits, labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/torchvision")
    parser.add_argument("--checkpoint", default="data/checkpoints/mnist_seed0.pt")
    parser.add_argument("--eval-count", type=int, default=500)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--rhos", nargs="+", type=float,
                        default=[0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75])
    parser.add_argument("--seed", type=int, default=143)
    parser.add_argument("--out", default="results/vision/model_correlation.csv")
    args = parser.parse_args()

    _, test, model = make_data("mnist", args.data)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    model.eval()
    rng = np.random.default_rng(args.seed)
    indices = rng.choice(len(test), size=min(args.eval_count, len(test)), replace=False)
    loader = DataLoader(Subset(test, indices.tolist()), batch_size=256,
                        shuffle=False, num_workers=0)
    batches = list(loader)
    x = torch.cat([b[0] for b in batches])
    labels = torch.cat([b[1] for b in batches])
    fp = logits_for(model, x, 256, torch.device("cpu"))
    fp_acc = accuracy(fp, labels)
    encoded, metadata_bytes, n_weights = encode_model_spw(model)
    rows = []
    for rho in args.rhos:
        logits, acc = run_rho(model, encoded, x, labels, args.samples, rho,
                              args.seed + int(rho * 10000))
        rows.append({
            "dataset": "mnist", "seed": args.seed, "rho": rho,
            "samples": args.samples,
            "theoretical_effective_samples": theoretical_effective_samples(args.samples, rho),
            "accuracy": acc, "accuracy_delta_fp32": acc - fp_acc,
            "logit_rmse_fp32": float(torch.mean((logits - fp) ** 2).sqrt()),
            "pbits": n_weights, "range_metadata_bytes": metadata_bytes,
        })
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"FP32={fp_acc:.4f}; wrote {out}")
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
