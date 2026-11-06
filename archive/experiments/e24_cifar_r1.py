from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from experiments.e06_vision import (accuracy, encode_model_spw, logits_for,
                                    make_data, quantize_float_model, quantize_model)


def main():
    train, test, model = make_data("cifar10", "data/torchvision")
    model.load_state_dict(torch.load("data/checkpoints/cifar10_seed0.pt",
                                    map_location="cpu", weights_only=True))
    indices = np.random.default_rng(20261026).choice(len(test), 2000, replace=False)
    loader = DataLoader(Subset(test, indices.tolist()), batch_size=256, shuffle=False)
    batches = list(loader)
    x = torch.cat([item[0] for item in batches])
    labels = torch.cat([item[1] for item in batches])
    device = torch.device("cpu")
    fp = logits_for(model, x, 256, device)
    fp_acc = accuracy(fp, labels)
    rows = []

    def add(name, logits, samples=0, metadata=0):
        rows.append({"dataset": "cifar10", "seed": 0, "representation": name,
                     "samples": samples, "accuracy": accuracy(logits, labels),
                     "accuracy_delta": accuracy(logits, labels) - fp_acc,
                     "logit_rmse": float(torch.mean((logits-fp)**2).sqrt()),
                     "pbits_per_weight": 1.0 if name.startswith("SPW") else 0.0,
                     "metadata_bytes": metadata})

    add("FP32", fp)
    for name, dtype in (("FP16", torch.float16), ("BF16", torch.bfloat16)):
        quant = quantize_float_model(model, dtype)
        add(name, logits_for(quant, x, 256, device))
    for bits in (8, 4, 2):
        quant = quantize_model(model, bits)
        add(f"INT{bits}", logits_for(quant, x, 256, device))

    encoded, metadata, _ = encode_model_spw(model)
    modules = dict(model.named_modules())
    originals = {name: modules[name].weight.detach().clone() for name, _, _ in encoded}
    sums = {name: np.zeros(tuple(modules[name].weight.shape), dtype=np.float64)
            for name, _, _ in encoded}
    rng = np.random.default_rng(991)
    budgets = {1, 4, 16, 64, 256}
    for t in range(1, max(budgets) + 1):
        for name, codec, state in encoded:
            sums[name] += codec.sample(state, 1, rng)[0]
        if t in budgets:
            for name, total in sums.items():
                modules[name].weight.data.copy_(torch.from_numpy(total / t).to(
                    dtype=modules[name].weight.dtype))
            add("SPW_reconstruct_then_infer", logits_for(model, x, 256, device),
                t, metadata)
    for name, old in originals.items():
        modules[name].weight.data.copy_(old)
    out = Path("results/vision/cifar_r1.csv")
    with out.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"FP32={fp_acc:.4f}; saved {out}")
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
