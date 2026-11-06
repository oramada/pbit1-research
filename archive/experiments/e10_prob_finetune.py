"""Probability-parameter fine-tuning for the MNIST MLP.

Weights are parameterized through one Bernoulli probability each, with fixed
per-output-channel centers/radii. The regularizer penalizes the expected sampled
weight variance. A separate sweep reconstructs weights from T p-bit draws.
"""
from __future__ import annotations

import argparse
import copy
import csv
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.func import functional_call
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from models.tiny_vision import MLP


def setup_probability_params(model: nn.Module, eps: float):
    weights = {}
    params = nn.ParameterDict()
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            pname = f"{name}.weight"
            w = dict(model.named_parameters())[pname].detach().cpu().numpy().astype(np.float64)
            lo = w.min(axis=1, keepdims=True)
            hi = w.max(axis=1, keepdims=True)
            center = (lo + hi) / 2.0
            radius = (hi - lo) / 2.0
            p = np.full_like(w, 0.5) if np.all(radius == 0) else 0.5 * ((w-center) / np.where(radius > 0, radius, 1.0) + 1.0)
            p = np.clip(p, eps, 1.0-eps)
            theta = np.log(p) - np.log1p(-p)
            key = f"param_{len(weights)}"
            params[key] = nn.Parameter(torch.tensor(theta, dtype=torch.float32))
            weights[pname] = {
                "key": key,
                "center": torch.tensor(center, dtype=torch.float32),
                "radius": torch.tensor(radius, dtype=torch.float32),
            }
    return params, weights


def functional_params(model, theta_params, specs):
    result = {name: p for name, p in model.named_parameters()}
    total_variance = torch.zeros((), dtype=torch.float32)
    count = 0
    for pname, spec in specs.items():
        theta = theta_params[spec["key"]]
        p = torch.sigmoid(theta)
        center, radius = spec["center"], spec["radius"]
        result[pname] = center + radius * (2.0 * p - 1.0)
        total_variance = total_variance + (4.0 * radius.square() * p * (1.0-p)).sum()
        count += p.numel()
    return result, total_variance / max(count, 1)


@torch.no_grad()
def evaluate_functional(model, theta_params, specs, loader, limit_batches=None):
    params, variance = functional_params(model, theta_params, specs)
    total = correct = loss_sum = 0.0
    criterion = nn.CrossEntropyLoss(reduction="sum")
    for batch_idx, (x, y) in enumerate(loader):
        if limit_batches is not None and batch_idx >= limit_batches:
            break
        logits = functional_call(model, params, (x,))
        loss_sum += float(criterion(logits, y))
        correct += float((logits.argmax(1) == y).sum())
        total += len(y)
    return correct / total, loss_sum / total, float(variance)


def train_probability_model(model, theta_params, specs, train_loader, epochs,
                            variance_weight, learning_rate):
    optimizer = torch.optim.Adam(theta_params.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    losses = []
    model.eval()
    for epoch in range(epochs):
        total_loss = total_n = 0
        for x, y in train_loader:
            optimizer.zero_grad(set_to_none=True)
            params, variance = functional_params(model, theta_params, specs)
            logits = functional_call(model, params, (x,))
            task = criterion(logits, y)
            loss = task + variance_weight * variance
            loss.backward()
            optimizer.step()
            total_loss += float(task.detach()) * len(y)
            total_n += len(y)
        losses.append(total_loss / total_n)
        print(f"lambda_var={variance_weight:g} epoch={epoch+1}/{epochs} task_loss={losses[-1]:.4f}", flush=True)
    return losses


def reconstruct_sampled(model, theta_params, specs, n_samples, seed):
    params, _ = functional_params(model, theta_params, specs)
    rng = np.random.default_rng(seed)
    reconstructed = {name: value.detach().clone() for name, value in params.items()}
    for pname, spec in specs.items():
        p = torch.sigmoid(theta_params[spec["key"]]).detach().cpu().numpy()
        c = spec["center"].numpy()
        r = spec["radius"].numpy()
        # Binomial counts are equivalent to n independent Bernoulli draws per weight.
        ones = rng.binomial(n_samples, p)
        mean_spin = 2.0 * ones / n_samples - 1.0
        reconstructed[pname] = torch.from_numpy(c + r * mean_spin).float()
    return reconstructed


@torch.no_grad()
def evaluate_parameters(model, params, loader):
    correct = total = loss = 0.0
    criterion = nn.CrossEntropyLoss(reduction="sum")
    for x, y in loader:
        logits = functional_call(model, params, (x,))
        loss += float(criterion(logits, y))
        correct += float((logits.argmax(1) == y).sum())
        total += len(y)
    return correct/total, loss/total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/torchvision")
    parser.add_argument("--checkpoint", default="data/checkpoints/mnist_seed0.pt")
    parser.add_argument("--out", default="results/vision/probability_finetune.csv")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--variance-weights", nargs="+", type=float, default=[0.0, 0.25, 1.0])
    parser.add_argument("--samples", nargs="+", type=int, default=[1, 4, 16, 64, 256, 1024])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    train = MNIST(args.data, train=True, download=True, transform=ToTensor())
    test = MNIST(args.data, train=False, download=True, transform=ToTensor())
    g = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, generator=g)
    test_loader = DataLoader(test, batch_size=512, shuffle=False)
    base = MLP()
    base.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    fp_acc, fp_loss, _ = evaluate_functional(base, nn.ParameterDict(), {}, test_loader)
    print(f"FP32 reference: accuracy={fp_acc:.4f}, CE={fp_loss:.4f}", flush=True)
    rows = [{"condition": "FP32_reference", "lambda_variance": 0.0, "samples": 0,
             "accuracy": fp_acc, "accuracy_delta": 0.0, "cross_entropy": fp_loss,
             "mean_weight_variance": 0.0}]
    for weight in args.variance_weights:
        # Reinitialize theta/range from the same pretrained model for paired comparison.
        model = copy.deepcopy(base)
        theta_params, specs = setup_probability_params(model, 1e-4)
        if args.epochs > 0:
            train_probability_model(model, theta_params, specs, train_loader,
                                    args.epochs, weight, 1e-4)
        accuracy_mean, loss_mean, variance_mean = evaluate_functional(
            model, theta_params, specs, test_loader)
        rows.append({"condition": "expected_weight_finetuned", "lambda_variance": weight,
                     "samples": 0, "accuracy": accuracy_mean,
                     "accuracy_delta": accuracy_mean-fp_acc, "cross_entropy": loss_mean,
                     "mean_weight_variance": variance_mean})
        for samples in args.samples:
            sampled = reconstruct_sampled(model, theta_params, specs, samples,
                                          args.seed + 1000 + samples)
            acc, loss = evaluate_parameters(model, sampled, test_loader)
            rows.append({"condition": "SPW_weight_reconstruction", "lambda_variance": weight,
                         "samples": samples, "accuracy": acc,
                         "accuracy_delta": acc-fp_acc, "cross_entropy": loss,
                         "mean_weight_variance": variance_mean})
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
