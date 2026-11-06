from __future__ import annotations

import argparse
import copy
import csv
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from models.tiny_vision import CIFARSmallCNN, MLP, SmallCNN
from weight_encodings.single_pbit import SinglePBit


def make_data(name, root):
    if name in ("mnist", "fashion", "fashion_mlp"):
        cls = datasets.MNIST if name == "mnist" else datasets.FashionMNIST
        transform = transforms.ToTensor()
        train = cls(root, train=True, download=True, transform=transform)
        test = cls(root, train=False, download=True, transform=transform)
        model = MLP() if name in ("mnist", "fashion_mlp") else SmallCNN()
    elif name == "cifar10":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(
            (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
        train = datasets.CIFAR10(root, train=True, download=True, transform=transform)
        test = datasets.CIFAR10(root, train=False, download=True, transform=transform)
        model = CIFARSmallCNN()
    else:
        raise ValueError(name)
    return train, test, model


def train_model(model, train_set, seed, epochs, batch_size, device):
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0,
                        generator=generator)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    losses = []
    for epoch in range(epochs):
        model.train()
        total_loss = total = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(y)
            total += len(y)
        losses.append(total_loss / total)
        print(f"seed={seed} epoch={epoch+1}/{epochs} loss={losses[-1]:.4f}", flush=True)
    return losses


@torch.no_grad()
def logits_for(model, x, batch_size, device):
    model.eval()
    out = []
    for start in range(0, len(x), batch_size):
        out.append(model(x[start:start+batch_size].to(device)).cpu())
    return torch.cat(out, dim=0)


def accuracy(logits, labels):
    return float((logits.argmax(1) == labels).float().mean())


def quantize_model(model, bits):
    result = copy.deepcopy(model)
    levels = (1 << (bits - 1)) - 1
    with torch.no_grad():
        for module in result.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                scale = module.weight.abs().max() / levels
                if float(scale) > 0:
                    if bits == 2:
                        scale = module.weight.abs().max() / 2
                        q = torch.round(module.weight / scale).clamp(-2, 1)
                    else:
                        q = torch.round(module.weight / scale).clamp(-levels, levels)
                    module.weight.copy_(q * scale)
    return result


def quantize_float_model(model, dtype):
    result = copy.deepcopy(model)
    with torch.no_grad():
        for parameter in result.parameters():
            parameter.copy_(parameter.to(dtype).to(parameter.dtype))
    return result


def encode_model_spw(model):
    encoded, metadata_bytes, count = [], 0, 0
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            weight = module.weight.detach().cpu().numpy().astype(np.float64)
            ids = np.broadcast_to(np.arange(weight.shape[0]).reshape(
                (-1,) + (1,) * (weight.ndim - 1)), weight.shape)
            codec = SinglePBit()
            state = codec.encode(weight, ids)
            encoded.append((name, codec, state))
            metadata_bytes += state.n_groups * 8
            count += weight.size
    return encoded, metadata_bytes, count


def stochastic_logits_curve(model, encoded, x, batch_size, device, sample_counts, seed):
    rng = np.random.default_rng(seed)
    modules = dict(model.named_modules())
    originals = {name: modules[name].weight.detach().clone() for name, _, _ in encoded}
    requested = set(sample_counts)
    maximum = max(requested)
    running = None
    stochastic_snapshots = {}
    reconstructed_snapshots = {}
    weight_sums = {name: np.zeros(tuple(module.weight.shape), dtype=np.float64)
                   for name, module in modules.items() if name in originals}
    for t in range(1, maximum + 1):
        for name, codec, state in encoded:
            sample = codec.sample(state, 1, rng)[0]
            weight_sums[name] += sample
            modules[name].weight.data.copy_(torch.from_numpy(sample).to(
                device=device, dtype=modules[name].weight.dtype))
        current = logits_for(model, x, batch_size, device)
        running = current if running is None else running + current
        if t in requested:
            stochastic_snapshots[t] = running / t
            # Reconstruct each parameter from T samples, then run the deterministic net.
            for name, total in weight_sums.items():
                modules[name].weight.data.copy_(torch.from_numpy(total / t).to(
                    device=device, dtype=modules[name].weight.dtype))
            reconstructed_snapshots[t] = logits_for(model, x, batch_size, device)
    for name, old in originals.items():
        modules[name].weight.data.copy_(old)
    return stochastic_snapshots, reconstructed_snapshots


def run_one(name, seed, args):
    print(f"loading {name}", flush=True)
    # Seed before model construction so initialization and data order share a seed.
    torch.manual_seed(seed)
    np.random.seed(seed)
    train, test, model = make_data(name, args.data)
    device = torch.device("cpu")
    torch.set_num_threads(args.threads)
    checkpoint = Path(args.checkpoints) / f"{name}_seed{seed}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if checkpoint.exists() and not args.retrain:
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        losses = []
        print(f"loading checkpoint {checkpoint}", flush=True)
    else:
        losses = train_model(model, train, seed, args.epochs, args.batch_size, device)
        torch.save(model.state_dict(), checkpoint)
    # One fixed evaluation subset across checkpoints enables paired seed comparisons.
    rng = np.random.default_rng(20261026)
    indices = rng.choice(len(test), size=min(args.eval_count, len(test)), replace=False)
    loader = DataLoader(Subset(test, indices.tolist()), batch_size=args.batch_size,
                        shuffle=False, num_workers=0)
    xb, yb = zip(*[(x, y) for x, y in loader])
    x, labels = torch.cat(xb), torch.cat(yb)
    fp = logits_for(model, x, args.batch_size, device)
    fp_acc = accuracy(fp, labels)
    print(f"{name} seed={seed} FP32 accuracy={fp_acc:.4f}", flush=True)
    rows = []
    def add(rep, logits, samples=0, pbits=0, metadata=0, avg_samples=0):
        rows.append({"dataset": name, "seed": seed, "representation": rep,
                     "samples": samples, "accuracy": accuracy(logits, labels),
                     "accuracy_delta": accuracy(logits, labels) - fp_acc,
                     "logit_rmse": float(torch.mean((logits - fp) ** 2).sqrt()),
                     "avg_samples_per_example": avg_samples, "pbits_per_weight": pbits,
                     "metadata_bytes": metadata,
                     "train_last_loss": losses[-1] if losses else float("nan")})
    add("FP32", fp)
    for label, dtype in (("FP16", torch.float16), ("BF16", torch.bfloat16)):
        logits = logits_for(quantize_float_model(model, dtype), x, args.batch_size, device)
        add(label, logits)
    for bits in (8, 4, 2):
        logits = logits_for(quantize_model(model, bits), x, args.batch_size, device)
        add(f"INT{bits}", logits)
    multi4 = logits_for(quantize_model(model, 4), x, args.batch_size, device)
    add("multi_pbit_K4_quantized", multi4, 1, 4, 0, 1)
    encoded, metadata, n_weights = encode_model_spw(model)
    curve, reconstructed = stochastic_logits_curve(model, encoded, x, args.batch_size, device,
                                                   args.sample_counts, seed + 991)
    for samples in args.sample_counts:
        logits = curve[samples]
        add("SPW_per_output_channel", logits, samples, 1, metadata, samples)
        add("SPW_reconstruct_then_infer", reconstructed[samples], samples, 1,
            metadata, samples)
    print(f"completed {name} seed={seed}; rows={len(rows)}; weights={n_weights}", flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["mnist", "fashion"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-count", type=int, default=2000)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--sample-counts", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--data", default="data/torchvision")
    parser.add_argument("--checkpoints", default="data/checkpoints")
    parser.add_argument("--out", default="results/vision/metrics.csv")
    parser.add_argument("--retrain", action="store_true")
    args = parser.parse_args()
    rows = []
    start = time.time()
    for name in args.datasets:
        for seed in args.seeds:
            rows.extend(run_one(name, seed, args))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved {out}, {len(rows)} rows, elapsed {time.time()-start:.1f}s")


if __name__ == "__main__":
    main()
