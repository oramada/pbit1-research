"""Larger learned [0,1] weights on a second dataset; conventional training only.

The signed control releases the projection from exactly the same initialization
and minibatch sequence. Validation CE selects checkpoints; test labels do not.
Native THRML inference is performed separately in its pinned environment.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import FashionMNIST


class FashionMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Linear(196, 64)
        self.second = nn.Linear(64, 10)
        nn.init.uniform_(self.first.weight, 0, .08*np.sqrt(64/196))
        nn.init.uniform_(self.second.weight, 0, .15*np.sqrt(32/64))
        nn.init.zeros_(self.first.bias)
        nn.init.zeros_(self.second.bias)

    def forward(self, x):
        return self.second(F.relu(self.first(x)))

    def project(self):
        with torch.no_grad():
            self.first.weight.clamp_(0, 1)
            self.second.weight.clamp_(0, 1)


def features(images):
    return F.adaptive_avg_pool2d(images.unsqueeze(1).float()/255, (14, 14)).flatten(1).contiguous()


def main():
    torch.set_num_threads(8)
    torch.use_deterministic_algorithms(True)
    out = Path("data/thrml_spw/positive_fashion")
    results = Path("results/thrml_spw/fashion_training")
    out.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    protocol = {"dataset": "FashionMNIST", "training": 55000, "validation": 5000, "test": 10000,
                "architecture": "196->64->10, ReLU; 13,184 matrix weights and74 deterministic biases",
                "preprocessing": "28x28 to14x14 adaptive average pooling, then scalar mean/std from the 55,000 training examples only",
                "seeds": list(range(5)), "epochs": 20, "batch_size": 256,
                "optimizer": "Adam lr0.003 with cosine schedule to0.0003 over20 epochs",
                "conditions": ["positive_projected", "signed_released_control"],
                "paired_design": "Same positive initialization and identical minibatch permutation per seed; only matrix projection differs",
                "selection": "Lowest validation CE checkpoint across20 epochs per condition; no test evaluation until checkpoint frozen",
                "test_labels_used_for_selection": False, "split_seed": 20261029,
                "scope": "Conventional deterministic training/export. No PyTorch stochastic weights or substitute sampler."}
    (results/"protocol.json").write_text(json.dumps(protocol, indent=2)+"\n")
    train = FashionMNIST("data/torchvision", train=True, download=False)
    test = FashionMNIST("data/torchvision", train=False, download=False)
    raw, raw_test = features(train.data), features(test.data)
    ids = np.random.default_rng(protocol["split_seed"]).permutation(len(train.targets))
    train_ids, val_ids = ids[:55000], ids[55000:]
    mean, std = raw[train_ids].mean().item(), raw[train_ids].std(unbiased=False).item()
    x, xt = ((raw-mean)/std).contiguous(), ((raw_test-mean)/std).contiguous()
    y, yt = train.targets, test.targets
    np.savez_compressed(out/"evaluation_data.npz", test_inputs=xt.numpy(), test_labels=yt.numpy(),
        validation_inputs=x[val_ids].numpy(), validation_labels=y[val_ids].numpy(),
        train_indices=train_ids, validation_indices=val_ids,
        normalization_mean=np.array(mean), normalization_std=np.array(std))
    histories, models, start = [], [], time.monotonic()
    for seed in range(5):
        torch.manual_seed(seed)
        initial = FashionMLP().state_dict()
        initial_digest = hashlib.sha256(b"".join(v.numpy().tobytes() for v in initial.values())).hexdigest()
        for condition in protocol["conditions"]:
            model = FashionMLP()
            model.load_state_dict(copy.deepcopy(initial))
            loader = DataLoader(TensorDataset(x[train_ids], y[train_ids]), batch_size=256, shuffle=True,
                                generator=torch.Generator().manual_seed(seed))
            opt = torch.optim.Adam(model.parameters(), lr=.003)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=20, eta_min=.0003)
            best_ce, best_epoch, best = np.inf, None, None
            for epoch in range(20):
                total = 0.
                model.train()
                for xb, yb in loader:
                    opt.zero_grad(set_to_none=True)
                    loss = F.cross_entropy(model(xb), yb)
                    loss.backward()
                    opt.step()
                    if condition == "positive_projected":
                        model.project()
                    total += loss.item()*len(yb)
                model.eval()
                with torch.no_grad():
                    val = model(x[val_ids])
                    ce = F.cross_entropy(val, y[val_ids]).item()
                    accuracy = (val.argmax(1) == y[val_ids]).float().mean().item()
                histories.append({"seed": seed, "condition": condition, "epoch": epoch+1,
                                  "train_ce": total/55000, "validation_ce": ce, "validation_accuracy": accuracy})
                if ce < best_ce:
                    best_ce, best_epoch, best = ce, epoch+1, copy.deepcopy(model.state_dict())
                scheduler.step()
            model.load_state_dict(best)
            model.eval()
            with torch.no_grad():
                logits, validation_logits = model(xt).numpy(), model(x[val_ids]).numpy()
            arrays = {"weight_0": model.first.weight.detach().numpy().T,
                      "weight_1": model.second.weight.detach().numpy().T,
                      "bias_0": model.first.bias.detach().numpy(), "bias_1": model.second.bias.detach().numpy(),
                      "reference_test_logits": logits, "reference_validation_logits": validation_logits}
            if condition == "positive_projected":
                assert all(np.all((arrays[f"weight_{i}"] >= 0) & (arrays[f"weight_{i}"] <= 1)) for i in range(2))
            np.savez_compressed(out/f"seed{seed}_{condition}.npz", **arrays)
            record = {"seed": seed, "condition": condition, "selected_epoch": best_epoch,
                      "selected_validation_ce": best_ce,
                      "test_accuracy": float(np.mean(logits.argmax(1) == yt.numpy())),
                      "matrix_parameters": 13184, "deterministic_neural_biases": 74,
                      "initial_state_sha256": initial_digest}
            models.append(record)
            print({**record, "elapsed_seconds": time.monotonic()-start}, flush=True)
    (results/"history.json").write_text(json.dumps(histories, indent=2)+"\n")
    manifest = {"config": protocol, "torch_version": torch.__version__, "normalization": {"mean": mean, "std": std},
                "elapsed_seconds": time.monotonic()-start, "models": models,
                "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "dataset_raw_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sorted(Path("data/torchvision/FashionMNIST/raw").glob("*-ubyte"))}}
    (results/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")


if __name__ == "__main__":
    main()
