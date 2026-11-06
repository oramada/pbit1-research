from __future__ import annotations

import argparse
import csv
from itertools import product
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from experiments.e06_vision import accuracy, logits_for, make_data
from weight_encodings.single_pbit import SinglePBit


SAMPLE_LEVELS = (1, 4, 16, 64, 256)
NO_SPW = 0


def read_curves(paths: list[Path]):
    rows = []
    for path in paths:
        if not path.exists():
            continue
        with path.open(newline="") as stream:
            rows.extend(csv.DictReader(stream))
    curves = {}
    for row in rows:
        key = (row["dataset"], row["layer"], int(row["samples"]))
        curves[key] = {
            "nweights": int(row["weight_count"]),
            "metadata": int(row["range_metadata_bytes"]),
            "loss_delta": float(row["cross_entropy_delta"]),
        }
    datasets = sorted({key[0] for key in curves})
    return curves, datasets


def optimize_one_dataset(dataset, curves, budget_fraction, max_loss_delta):
    layers = sorted({layer for name, layer, t in curves if name == dataset})
    if not layers:
        raise ValueError(f"no layer curves for {dataset}")
    full64 = sum(curves[(dataset, layer, 64)]["nweights"] * 64 for layer in layers)
    budget = int(round(full64 * budget_fraction))
    options = []
    for layer in layers:
        layer_options = [(0, 0, 0.0, 0)]  # FP32 layer, no p-bit transitions
        for t in SAMPLE_LEVELS:
            key = (dataset, layer, t)
            if key not in curves:
                continue
            data = curves[key]
            layer_options.append((t, data["nweights"] * t,
                                  data["loss_delta"], data["metadata"]))
        options.append(layer_options)
    candidates = []
    for assignment in product(*options):
        transitions = sum(item[1] for item in assignment)
        if transitions > budget:
            continue
        pred_delta = sum(item[2] for item in assignment)
        converted_count = sum(curves[(dataset, layers[i], item[0])]["nweights"]
                              for i, item in enumerate(assignment) if item[0] > 0)
        if pred_delta <= max_loss_delta:
            candidates.append((converted_count, pred_delta, transitions, assignment))
    # Maximize converted parameters under the transition budget and a quality bound.
    if not candidates:
        raise RuntimeError(f"No allocation fits CE tolerance {max_loss_delta}")
    converted_count, pred_delta, transitions, allocation = max(
        candidates, key=lambda x: (x[0], -x[1], -x[2]))
    selected = {layer: allocation[i][0] for i, layer in enumerate(layers)}
    return {"layers": layers, "selected": selected, "budget": budget,
            "full64": full64, "transitions": transitions,
            "predicted_loss_delta": pred_delta,
            "converted_weight_count_proxy": converted_count,
            "metadata": sum(allocation[i][3] for i in range(len(layers))),
            "converted_weights": sum(curves[(dataset, layer, selected[layer])]["nweights"]
                                     for layer in layers if selected[layer] > 0)}


def evaluate_reconstructed(dataset, selected, seed, args):
    _, test, model = make_data(dataset, args.data)
    model.load_state_dict(torch.load(Path(args.checkpoints) / f"{dataset}_seed{seed}.pt",
                                     map_location="cpu", weights_only=True))
    model.eval()
    rng_idx = np.random.default_rng(seed + 1000)
    indices = rng_idx.choice(len(test), size=min(args.eval_count, len(test)), replace=False)
    loader = DataLoader(Subset(test, indices.tolist()), batch_size=args.batch_size,
                        shuffle=False, num_workers=0)
    pieces = list(loader)
    x = torch.cat([item[0] for item in pieces])
    labels = torch.cat([item[1] for item in pieces])
    fp_logits = logits_for(model, x, args.batch_size, torch.device("cpu"))
    fp_acc = accuracy(fp_logits, labels)
    modules = dict(model.named_modules())
    rng = np.random.default_rng(seed + 50122)
    metadata, converted_n = 0, 0
    for name, n_samples in selected.items():
        if n_samples == NO_SPW:
            continue
        module = modules[name]
        w = module.weight.detach().numpy().astype(np.float64)
        gids = np.broadcast_to(np.arange(w.shape[0]).reshape((-1,) + (1,) * (w.ndim - 1)), w.shape)
        codec = SinglePBit(eps=1e-6)
        encoded = codec.encode(w, gids)
        draws = codec.sample(encoded, n_samples, rng)
        reconstructed = draws.mean(axis=0)
        module.weight.data.copy_(torch.from_numpy(reconstructed).to(dtype=module.weight.dtype))
        metadata += encoded.n_groups * 8
        converted_n += w.size
    reconstructed_logits = logits_for(model, x, args.batch_size, torch.device("cpu"))
    ce_fp = float(nn.functional.cross_entropy(fp_logits, labels))
    ce_recon = float(nn.functional.cross_entropy(reconstructed_logits, labels))
    return {"fp_accuracy": fp_acc,
            "accuracy": accuracy(reconstructed_logits, labels),
            "accuracy_delta": accuracy(reconstructed_logits, labels) - fp_acc,
            "cross_entropy": ce_recon,
            "cross_entropy_delta": ce_recon - ce_fp,
            "logit_rmse": float(torch.mean((reconstructed_logits - fp_logits) ** 2).sqrt()),
            "metadata_bytes": metadata, "converted_weights": converted_n}


def main():
    parser = argparse.ArgumentParser(description="Sensitivity-guided hybrid SPW allocation.")
    parser.add_argument("--sensitivity", nargs="+", default=[
        "results/sensitivity/layer_sensitivity.csv",
        "results/sensitivity_t256/layer_sensitivity.csv"])
    parser.add_argument("--datasets", nargs="+", default=["mnist", "fashion"])
    parser.add_argument("--budget-fractions", nargs="+", type=float,
                        default=[0.125, 0.25, 0.5, 1.0, 2.0])
    parser.add_argument("--max-loss-delta", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-count", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--data", default="data/torchvision")
    parser.add_argument("--checkpoints", default="data/checkpoints")
    parser.add_argument("--out-dir", default="results/selection")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    paths = [Path(path) for path in args.sensitivity]
    curves, available_datasets = read_curves(paths)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    selected_rows = []
    for dataset in args.datasets:
        if dataset not in available_datasets:
            print(f"skip {dataset}: no sensitivity data", flush=True)
            continue
        for fraction in args.budget_fractions:
            plan = optimize_one_dataset(dataset, curves, fraction, args.max_loss_delta)
            measured = evaluate_reconstructed(dataset, plan["selected"], args.seed, args)
            row = {"dataset": dataset, "budget_fraction_of_all_layers_T64": fraction,
                   "budget_transitions": plan["budget"],
                   "selected_transitions": plan["transitions"],
                   "transition_budget_utilization": plan["transitions"] / max(1, plan["budget"]),
                   "converted_weights": measured["converted_weights"],
                   "metadata_bytes": measured["metadata_bytes"],
                   "predicted_additive_ce_delta_from_single_layer_curve": plan["predicted_loss_delta"],
                   **{f"T__{name}": value for name, value in plan["selected"].items()},
                   **{f"measured_{key}": value for key, value in measured.items()}}
            rows.append(row)
            selected_rows.append({"dataset": dataset, "budget_fraction": fraction,
                                  "allocation": ";".join(f"{n}={t}" for n, t in plan["selected"].items()),
                                  "transitions": plan["transitions"],
                                  "predicted_ce_delta": plan["predicted_loss_delta"],
                                  "measured_accuracy": measured["accuracy"],
                                  "measured_accuracy_delta": measured["accuracy_delta"],
                                  "measured_ce_delta": measured["cross_entropy_delta"],
                                  "logit_rmse": measured["logit_rmse"]})
            print(f"{dataset} budget={fraction:.3g}x allocation={plan['selected']} "
                  f"actual_acc={measured['accuracy']:.4f} actual_dCE={measured['cross_entropy_delta']:+.4f}",
                  flush=True)
    if not rows:
        raise RuntimeError("No model allocations were evaluated")
    for filename, data in (("allocations.csv", rows), ("comparison.csv", selected_rows)):
        path = out_dir / filename
        fieldnames = list(dict.fromkeys(key for item in data for key in item))
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(data)
    caveats = """# Sensitivity-guided hybrid SPW allocation

The solver enumerates, for each layer, FP32 (T=0) or one-p-bit-per-weight SPW with
T in {1, 4, 16, 64, 256}. It maximizes the fraction of weights converted subject to
an additive cross-entropy bound (default 0.10) and a fixed transition budget. It uses
one-layer cross-entropy deltas from `results/sensitivity/` and
`results/sensitivity_t256/`. Budgets are fractions of the cost of converting every layer at
T=64. Transition count is sum_l N_l T_l; probability metadata are separately reported.

This is a Phase 14/22 sensitivity proxy, not an optimality claim: one-layer loss
deltas need not add when layers are jointly noisy. The script therefore validates
each chosen hybrid configuration by reconstructing each selected layer's weights
from its own T samples and running the full deterministic network. The selection
proxy measures a different execution mode (averaged logits from repeated full-network
stochastic passes), so predicted versus measured cross-entropy can disagree. The
actual reconstruction results are in `allocations.csv` and `comparison.csv`.

Only SPW's measured K=1 option is optimized. K=2/4 quality curves were not measured
by the current layer sensitivity experiment, so assigning fabricated performance
to those options would be misleading. FP32 is the hybrid fallback.
"""
    (out_dir / "README.md").write_text(caveats)
    print(f"saved {out_dir}")


if __name__ == "__main__":
    main()
