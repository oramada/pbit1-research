"""Phase 12 Gumbel/Concrete and local-reparameterization training estimators."""
from __future__ import annotations

import copy
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.func import functional_call
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from models.tiny_vision import MLP

SEED = 1201
EPS = 1e-4


def prepare(model):
    theta = nn.ParameterDict()
    specs = {}
    named = dict(model.named_parameters())
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        pname = f"{name}.weight"
        w = named[pname].detach().numpy().astype(np.float64)
        lo, hi = w.min(axis=1, keepdims=True), w.max(axis=1, keepdims=True)
        c, r = (lo+hi)/2, (hi-lo)/2
        p = np.full_like(w, 0.5) if not np.any(r) else (
            0.5 * ((w-c)/np.where(r > 0, r, 1.0) + 1.0))
        p = np.clip(p, EPS, 1-EPS)
        key = f"theta_{len(specs)}"
        theta[key] = nn.Parameter(torch.from_numpy(np.log(p)-np.log1p(-p)).float())
        specs[pname] = {"key": key, "center": torch.from_numpy(c).float(),
                        "radius": torch.from_numpy(r).float()}
    return theta, specs


def moments(theta, specs):
    means, variances, probs = {}, {}, {}
    for pname, spec in specs.items():
        p = torch.sigmoid(theta[spec["key"]])
        means[pname] = spec["center"] + spec["radius"] * (2*p-1)
        variances[pname] = 4 * spec["radius"].square() * p * (1-p)
        probs[pname] = p
    return means, variances, probs


def concrete_parameters(model, theta, specs, tau, generator):
    params = {name: value for name, value in model.named_parameters()}
    for pname, spec in specs.items():
        p = torch.sigmoid(theta[spec["key"]])
        u = torch.rand(p.shape, generator=generator).clamp(EPS, 1-EPS)
        logistic = torch.log(u) - torch.log1p(-u)
        z = torch.sigmoid((theta[spec["key"]] + logistic) / tau)
        params[pname] = spec["center"] + spec["radius"] * (2*z-1)
    return params


def local_reparameterized_logits(model, x, theta, specs):
    means, variances, _ = moments(theta, specs)
    # MLP's Linear layers appear at these parameter paths. Sampling each activation
    # independently is the standard local-reparameterization approximation.
    h = x.flatten(1)
    linear_names = [name for name, module in model.named_modules() if isinstance(module, nn.Linear)]
    modules = dict(model.named_modules())
    for index, name in enumerate(linear_names):
        module = modules[name]
        pname = f"{name}.weight"
        mean = F.linear(h, means[pname], module.bias)
        variance = F.linear(h.square(), variances[pname], None).clamp_min(0)
        h = mean + torch.randn_like(mean) * torch.sqrt(variance + 1e-12)
        if index + 1 < len(linear_names):
            h = F.relu(h)
    return h


def train_one(model, theta, specs, loader, estimator, epochs=1, tau=0.5):
    optimizer = torch.optim.Adam(theta.parameters(), lr=5e-5)
    criterion = nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(SEED + (0 if estimator == "concrete" else 1))
    history = []
    model.eval()
    for epoch in range(epochs):
        loss_sum = total = 0
        for x, y in loader:
            optimizer.zero_grad(set_to_none=True)
            if estimator == "concrete":
                params = concrete_parameters(model, theta, specs, tau, generator)
                logits = functional_call(model, params, (x,))
            elif estimator == "local_reparameterization":
                logits = local_reparameterized_logits(model, x, theta, specs)
            else:
                raise ValueError(estimator)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(y)
            total += len(y)
        history.append(loss_sum/total)
        print(f"{estimator} epoch={epoch+1}/{epochs} stochastic_CE={history[-1]:.4f}", flush=True)
    return history


@torch.no_grad()
def evaluate(model, params, loader):
    correct = total = loss = 0.0
    for x, y in loader:
        logits = functional_call(model, params, (x,))
        loss += float(F.cross_entropy(logits, y, reduction="sum"))
        correct += float((logits.argmax(1) == y).sum())
        total += len(y)
    return correct/total, loss/total


def mean_parameters(model, theta, specs):
    params = {name: p for name, p in model.named_parameters()}
    means, _, _ = moments(theta, specs)
    params.update(means)
    return params


def reconstruct_parameters(model, theta, specs, samples, seed):
    rng = np.random.default_rng(seed)
    params = {name: p.detach().clone() for name, p in mean_parameters(model, theta, specs).items()}
    _, _, probs = moments(theta, specs)
    for pname, spec in specs.items():
        p = probs[pname].detach().numpy()
        spin_mean = 2*rng.binomial(samples, p)/samples - 1
        params[pname] = torch.from_numpy(spec["center"].numpy() +
                                         spec["radius"].numpy()*spin_mean).float()
    return params


def main():
    torch.manual_seed(SEED)
    torch.set_num_threads(8)
    train_set = MNIST("data/torchvision", train=True, download=True, transform=ToTensor())
    test_set = MNIST("data/torchvision", train=False, download=True, transform=ToTensor())
    train_loader = DataLoader(train_set, batch_size=256, shuffle=True,
                              generator=torch.Generator().manual_seed(SEED))
    test_idx = np.random.default_rng(SEED).choice(len(test_set), 2000, replace=False)
    test_loader = DataLoader(Subset(test_set, test_idx.tolist()), batch_size=512, shuffle=False)
    base = MLP()
    base.load_state_dict(torch.load("data/checkpoints/mnist_seed0.pt",
                                    map_location="cpu", weights_only=True))
    base_acc, base_loss = evaluate(base, dict(base.named_parameters()), test_loader)
    rows = [{"estimator": "FP32_checkpoint", "train_stochastic_CE": float("nan"),
             "inference": "deterministic", "samples": 0, "accuracy": base_acc,
             "accuracy_delta": 0.0, "cross_entropy": base_loss}]
    print(f"FP32 checkpoint accuracy={base_acc:.4f}", flush=True)
    for estimator in ("concrete", "local_reparameterization"):
        model = copy.deepcopy(base)
        theta, specs = prepare(model)
        history = train_one(model, theta, specs, train_loader, estimator, epochs=1)
        expected = mean_parameters(model, theta, specs)
        acc, loss = evaluate(model, expected, test_loader)
        rows.append({"estimator": estimator, "train_stochastic_CE": history[-1],
                     "inference": "expected_weights", "samples": 0, "accuracy": acc,
                     "accuracy_delta": acc-base_acc, "cross_entropy": loss})
        sampled = reconstruct_parameters(model, theta, specs, 64, SEED+64)
        acc64, loss64 = evaluate(model, sampled, test_loader)
        rows.append({"estimator": estimator, "train_stochastic_CE": history[-1],
                     "inference": "reconstructed_weights", "samples": 64,
                     "accuracy": acc64, "accuracy_delta": acc64-base_acc,
                     "cross_entropy": loss64})
    # Load STE results from the previous Phase 12 run for a direct paired reference.
    ste_path = Path("results/vision/monte_carlo_ste_3epoch.csv")
    if ste_path.exists():
        with ste_path.open() as stream:
            for row in csv.DictReader(stream):
                if row["condition"] in ("STE_expected_weights", "STE_SPW_reconstructed") and row["samples"] in ("0", "64"):
                    rows.append({"estimator": "straight_through_reference",
                                 "train_stochastic_CE": row["stochastic_train_CE"],
                                 "inference": row["condition"], "samples": int(row["samples"]),
                                 "accuracy": float(row["accuracy"]),
                                 "accuracy_delta": float(row["accuracy_delta"]),
                                 "cross_entropy": float(row["cross_entropy"])})
    out = Path("results/vision/phase12_estimators.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
