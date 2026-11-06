"""Phase 12: Monte Carlo weight training with a straight-through estimator."""
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


def initialize_probability_state(model: nn.Module, epsilon: float):
    theta = nn.ParameterDict()
    specs = {}
    named_params = dict(model.named_parameters())
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        pname = f"{name}.weight"
        weight = named_params[pname].detach().cpu().numpy().astype(np.float64)
        low = weight.min(axis=1, keepdims=True)
        high = weight.max(axis=1, keepdims=True)
        center = (low + high) / 2.0
        radius = (high - low) / 2.0
        p = np.full_like(weight, 0.5) if np.all(radius == 0) else (
            0.5 * ((weight-center) / np.where(radius > 0, radius, 1.0) + 1.0))
        p = np.clip(p, epsilon, 1.0-epsilon)
        key = f"prob_{len(specs)}"
        theta[key] = nn.Parameter(torch.from_numpy(np.log(p)-np.log1p(-p)).float())
        specs[pname] = {"key": key, "center": torch.from_numpy(center).float(),
                        "radius": torch.from_numpy(radius).float()}
    return theta, specs


def parameters_for(model, theta, specs, stochastic: bool, generator=None):
    params = {name: value for name, value in model.named_parameters()}
    probabilities = {}
    for pname, spec in specs.items():
        p = torch.sigmoid(theta[spec["key"]])
        expected = spec["center"] + spec["radius"] * (2.0*p - 1.0)
        if stochastic:
            bits = torch.bernoulli(p, generator=generator)
            draw = spec["center"] + spec["radius"] * (2.0*bits - 1.0)
            # Forward uses a Bernoulli weight draw; backward follows expected weight.
            weight = expected + (draw - expected).detach()
        else:
            weight = expected
        params[pname] = weight
        probabilities[pname] = p
    return params, probabilities


def train_ste(model, theta, specs, loader, epochs, lr, seed):
    optimizer = torch.optim.Adam(theta.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(seed)
    history = []
    model.eval()
    for epoch in range(epochs):
        total_loss = total = 0.0
        for x, y in loader:
            optimizer.zero_grad(set_to_none=True)
            params, _ = parameters_for(model, theta, specs, True, generator)
            logits = functional_call(model, params, (x,))
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(y)
            total += len(y)
        history.append(total_loss / total)
        print(f"STE epoch={epoch+1}/{epochs} stochastic_train_CE={history[-1]:.4f}", flush=True)
    return history


@torch.no_grad()
def evaluate(model, params, loader):
    correct = total = loss_sum = 0.0
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    for x, y in loader:
        logits = functional_call(model, params, (x,))
        loss_sum += float(loss_fn(logits, y))
        correct += float((logits.argmax(1) == y).sum())
        total += len(y)
    return correct/total, loss_sum/total


def reconstruct_parameters(model, theta, specs, samples, rng):
    params, _ = parameters_for(model, theta, specs, False)
    output = {name: p.detach().clone() for name, p in params.items()}
    for pname, spec in specs.items():
        p = torch.sigmoid(theta[spec["key"]]).detach().cpu().numpy()
        bits = rng.binomial(samples, p)
        spin_mean = 2.0 * bits / samples - 1.0
        output[pname] = torch.from_numpy(
            spec["center"].numpy() + spec["radius"].numpy() * spin_mean).float()
    return output


@torch.no_grad()
def monte_carlo_logits(model, theta, specs, loader, samples, seed, report_at):
    generator = torch.Generator().manual_seed(seed)
    running_logits = []
    labels = None
    for t in range(samples):
        params, _ = parameters_for(model, theta, specs, True, generator)
        one = []
        ys = []
        for x, y in loader:
            one.append(functional_call(model, params, (x,)))
            ys.append(y)
        logits = torch.cat(one)
        labels = torch.cat(ys)
        running_logits.append(logits)
        if t + 1 in report_at:
            mean = torch.stack(running_logits).mean(0)
            yield t + 1, mean, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/torchvision")
    parser.add_argument("--checkpoint", default="data/checkpoints/mnist_seed0.pt")
    parser.add_argument("--out", default="results/vision/monte_carlo_ste.csv")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--samples", nargs="+", type=int, default=[1, 16, 64, 256])
    parser.add_argument("--eval-count", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1201)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    torch.set_num_threads(8)
    train_data = MNIST(args.data, train=True, download=True, transform=ToTensor())
    test_data = MNIST(args.data, train=False, download=True, transform=ToTensor())
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(args.seed))
    rng = np.random.default_rng(args.seed)
    indices = rng.choice(len(test_data), min(args.eval_count, len(test_data)), replace=False)
    test_loader = DataLoader(torch.utils.data.Subset(test_data, indices.tolist()),
                             batch_size=512, shuffle=False)
    model = MLP()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    model.eval()
    empty_params = {name: p for name, p in model.named_parameters()}
    fp_acc, fp_loss = evaluate(model, empty_params, test_loader)
    theta, specs = initialize_probability_state(model, 1e-4)
    history = train_ste(model, theta, specs, train_loader, args.epochs,
                        args.learning_rate, args.seed)
    expected_params, probabilities = parameters_for(model, theta, specs, False)
    expected_acc, expected_loss = evaluate(model, expected_params, test_loader)
    rows = [
        {"condition": "FP32_checkpoint", "samples": 0, "accuracy": fp_acc,
         "accuracy_delta": 0.0, "cross_entropy": fp_loss,
         "stochastic_train_CE": float("nan"), "mean_weight_variance": float("nan")},
        {"condition": "STE_expected_weights", "samples": 0, "accuracy": expected_acc,
         "accuracy_delta": expected_acc-fp_acc, "cross_entropy": expected_loss,
         "stochastic_train_CE": history[-1] if history else float("nan"),
         "mean_weight_variance": float(np.mean([
             (4.0 * spec["radius"].numpy()**2 * p.detach().numpy() *
              (1.0-p.detach().numpy())).mean()
             for pname, spec in specs.items() for p in [probabilities[pname]]]))},
    ]
    for samples in sorted(set(args.samples)):
        sampled = reconstruct_parameters(model, theta, specs, samples,
                                         np.random.default_rng(args.seed + samples))
        acc, loss = evaluate(model, sampled, test_loader)
        rows.append({"condition": "STE_SPW_reconstructed", "samples": samples,
                     "accuracy": acc, "accuracy_delta": acc-fp_acc,
                     "cross_entropy": loss,
                     "stochastic_train_CE": history[-1] if history else float("nan"),
                     "mean_weight_variance": rows[1]["mean_weight_variance"]})
    max_samples = max(args.samples)
    for samples, logits, labels in monte_carlo_logits(
            model, theta, specs, test_loader, max_samples, args.seed + 9000,
            set(args.samples)):
        if samples in args.samples:
            acc = float((logits.argmax(1) == labels).float().mean())
            loss = float(nn.functional.cross_entropy(logits, labels))
            rows.append({"condition": "STE_SPW_direct_MC_logits", "samples": samples,
                         "accuracy": acc, "accuracy_delta": acc-fp_acc,
                         "cross_entropy": loss,
                         "stochastic_train_CE": history[-1] if history else float("nan"),
                         "mean_weight_variance": rows[1]["mean_weight_variance"]})
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"FP32={fp_acc:.4f} STE expected={expected_acc:.4f}; saved {out}", flush=True)


if __name__ == "__main__":
    main()
