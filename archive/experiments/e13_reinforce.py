from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.func import functional_call
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from models.tiny_vision import MLP
from weight_encodings.single_pbit import SinglePBit


def make_weight_parameters(model: nn.Module, eps: float):
    """Per-output-channel affine probability encoding for every linear weight."""
    groups = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            weight = module.weight.detach()
            lo = weight.amin(dim=1, keepdim=True)
            hi = weight.amax(dim=1, keepdim=True)
            center = (lo + hi) / 2
            radius = (hi - lo) / 2
            probability = torch.where(radius > 0,
                                      0.5 * ((weight - center) / radius + 1),
                                      torch.full_like(weight, 0.5))
            probability = probability.clamp(eps, 1.0 - eps)
            theta = nn.Parameter(torch.logit(probability))
            groups[name] = (theta, center, radius)
    return groups


def expected_weights(groups):
    return {f"{name}.weight": center + radius * (2 * theta.sigmoid() - 1)
            for name, (theta, center, radius) in groups.items()}


def sampled_weights(groups, generator, straight_through=False):
    replacements = {}
    samples = {}
    for name, (theta, center, radius) in groups.items():
        p = theta.sigmoid()
        z = torch.bernoulli(p, generator=generator)
        w_sample = center + radius * (2 * z - 1)
        samples[name] = (z, p)
        if straight_through:
            w_exp = center + radius * (2 * p - 1)
            w_sample = w_sample.detach() + w_exp - w_exp.detach()
        replacements[f"{name}.weight"] = w_sample
    return replacements, samples


def eval_logits(model, groups, x, batch_size, sample_count=0, seed=0):
    model.eval()
    outputs = []
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = x[start:start + batch_size]
            if sample_count <= 0:
                out = functional_call(model, expected_weights(groups), (xb,), strict=False)
            else:
                total = None
                for _ in range(sample_count):
                    weights, _ = sampled_weights(groups, generator, False)
                    logits = functional_call(model, weights, (xb,), strict=False)
                    total = logits if total is None else total + logits
                out = total / sample_count
            outputs.append(out)
    return torch.cat(outputs)


def accuracy(logits, labels):
    return float((logits.argmax(1) == labels).float().mean())


def train_method(model, initial_groups, train_loader, method, args, seed):
    groups = {}
    for name, (theta, center, radius) in initial_groups.items():
        groups[name] = (nn.Parameter(theta.detach().clone()), center, radius)
    optimizer = torch.optim.Adam([value[0] for value in groups.values()], lr=args.lr)
    generator = torch.Generator().manual_seed(seed + 1234)
    baseline = None
    losses, reinforce_grad_norms = [], []
    model.train()
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        n = 0
        for x, y in train_loader:
            optimizer.zero_grad(set_to_none=True)
            if method == "expected":
                weights = expected_weights(groups)
                logits = functional_call(model, weights, (x,), strict=False)
                loss = nn.functional.cross_entropy(logits, y)
                loss.backward()
            elif method == "ste":
                weights, _ = sampled_weights(groups, generator, True)
                logits = functional_call(model, weights, (x,), strict=False)
                loss = nn.functional.cross_entropy(logits, y)
                loss.backward()
            elif method == "reinforce":
                weights, samples = sampled_weights(groups, generator, False)
                logits = functional_call(model, weights, (x,), strict=False)
                loss = nn.functional.cross_entropy(logits, y)
                log_probability = torch.zeros((), dtype=torch.float32)
                for name, (z, p) in samples.items():
                    p = p.clamp(1e-7, 1 - 1e-7)
                    log_probability = log_probability + (z * torch.log(p) +
                                                           (1-z) * torch.log1p(-p)).sum()
                reward_advantage = loss.detach() if baseline is None else loss.detach() - baseline
                surrogate = reward_advantage * log_probability
                surrogate.backward()
                reinforce_grad_norms.append(float(torch.sqrt(sum(
                    torch.sum(value[0].grad ** 2) for value in groups.values()
                    if value[0].grad is not None))))
                current = float(loss.detach())
                baseline = current if baseline is None else (
                    args.baseline_decay * baseline + (1-args.baseline_decay) * current)
            else:
                raise ValueError(method)
            optimizer.step()
            epoch_loss += float(loss.detach()) * len(y)
            n += len(y)
        mean_loss = epoch_loss / n
        losses.append(mean_loss)
        print(f"{method} seed={seed} epoch={epoch+1}/{args.epochs} "
              f"sampled_train_ce={mean_loss:.4f} baseline={baseline}", flush=True)
    grad_mean = float(np.mean(reinforce_grad_norms)) if reinforce_grad_norms else 0.0
    return groups, losses, grad_mean


def main():
    parser = argparse.ArgumentParser(description="Compare expected, STE, and REINFORCE SPW training.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--train-count", type=int, default=10000)
    parser.add_argument("--eval-count", type=int, default=2000)
    parser.add_argument("--eval-samples", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--eps", type=float, default=1e-2)
    parser.add_argument("--baseline-decay", type=float, default=0.9)
    parser.add_argument("--data", default="data/torchvision")
    parser.add_argument("--checkpoint", default="data/checkpoints/mnist_seed0.pt")
    parser.add_argument("--out", default="results/reinforce/metrics.csv")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    train_set = datasets.MNIST(args.data, train=True, download=True,
                               transform=transforms.ToTensor())
    test_set = datasets.MNIST(args.data, train=False, download=True,
                              transform=transforms.ToTensor())
    rng = np.random.default_rng(args.seed)
    train_ids = rng.choice(len(train_set), size=min(args.train_count, len(train_set)),
                           replace=False)
    eval_ids = np.random.default_rng(args.seed + 7000).choice(
        len(test_set), size=min(args.eval_count, len(test_set)), replace=False)
    train_loader = DataLoader(Subset(train_set, train_ids.tolist()), batch_size=args.batch_size,
                              shuffle=True, num_workers=0,
                              generator=torch.Generator().manual_seed(args.seed))
    eval_loader = DataLoader(Subset(test_set, eval_ids.tolist()), batch_size=args.batch_size,
                             shuffle=False, num_workers=0)
    eval_batches = list(eval_loader)
    x = torch.cat([batch[0] for batch in eval_batches])
    y = torch.cat([batch[1] for batch in eval_batches])
    model = MLP()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    model.eval()
    with torch.no_grad():
        fp = torch.cat([model(xb) for xb in x.split(args.batch_size)])
    fp_ce = float(nn.functional.cross_entropy(fp, y))
    initial = make_weight_parameters(model, args.eps)
    rows = []
    for method in ("expected", "ste", "reinforce"):
        trained, train_losses, grad_norm = train_method(
            model, initial, train_loader, method, args, args.seed)
        expected = eval_logits(model, trained, x, args.batch_size)
        expected_ce = float(nn.functional.cross_entropy(expected, y))
        sampled = eval_logits(model, trained, x, args.batch_size,
                              sample_count=args.eval_samples,
                              seed=args.seed + 900)
        row = {"method": method, "seed": args.seed, "epochs": args.epochs,
               "train_count": len(train_ids), "eval_count": len(eval_ids),
               "initial_fp32_accuracy": accuracy(fp, y), "initial_fp32_ce": fp_ce,
               "last_sampled_train_ce": train_losses[-1],
               "expected_weight_accuracy": accuracy(expected, y),
               "expected_weight_ce": expected_ce,
               "expected_weight_ce_delta_from_fp32": expected_ce - fp_ce,
               "sampled_16_accuracy": accuracy(sampled, y),
               "sampled_16_ce": float(nn.functional.cross_entropy(sampled, y)),
               "reinforce_mean_grad_norm": grad_norm}
        rows.append(row)
        print(f"RESULT {method}: trainCE={row['last_sampled_train_ce']:.4f} "
              f"expectedAcc={row['expected_weight_accuracy']:.4f} "
              f"sampled16Acc={row['sampled_16_accuracy']:.4f} "
              f"expectedCE={expected_ce:.4f}", flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
