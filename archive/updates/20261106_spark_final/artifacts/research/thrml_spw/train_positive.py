"""Export genuinely learned w in [0,1] for native THRML inference.

Run in the existing vision environment. Native sampling is deliberately performed
by a separate script in the pinned THRML environment, not by PyTorch Bernoulli.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import MNIST


class PositiveMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.first=nn.Linear(64,32)
        self.second=nn.Linear(32,10)
        nn.init.uniform_(self.first.weight,0,.08)
        nn.init.uniform_(self.second.weight,0,.15)
        nn.init.zeros_(self.first.bias);nn.init.zeros_(self.second.bias)

    def forward(self,x):return self.second(F.relu(self.first(x)))

    def project(self):
        with torch.no_grad():
            self.first.weight.clamp_(0,1);self.second.weight.clamp_(0,1)


def features(data):
    x=data.unsqueeze(1).float()/255
    return ((F.adaptive_avg_pool2d(x,(8,8)).flatten(1)-.1307)/.3081).contiguous()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--seeds",type=int,nargs="+",default=list(range(5)))
    parser.add_argument("--epochs",type=int,default=10)
    args=parser.parse_args()
    torch.set_num_threads(8)
    out=Path("data/thrml_spw/positive_mnist");out.mkdir(parents=True,exist_ok=True)
    results=Path("results/thrml_spw/positive_training");results.mkdir(parents=True,exist_ok=True)
    train=MNIST("data/torchvision",train=True,download=False)
    test=MNIST("data/torchvision",train=False,download=False)
    x,y=features(train.data),train.targets
    xt,yt=features(test.data),test.targets
    split=np.random.default_rng(20261026).permutation(len(y))
    train_ids,val_ids=split[:55000],split[55000:]
    np.savez_compressed(out/"evaluation_data.npz",test_inputs=xt.numpy(),test_labels=yt.numpy(),
                        validation_inputs=x[val_ids].numpy(),validation_labels=y[val_ids].numpy(),
                        train_indices=train_ids,validation_indices=val_ids)
    metrics=[];start=time.monotonic()
    for seed in args.seeds:
        torch.manual_seed(seed)
        model=PositiveMLP()
        loader=DataLoader(TensorDataset(x[train_ids],y[train_ids]),batch_size=256,shuffle=True,
                          generator=torch.Generator().manual_seed(seed))
        optimizer=torch.optim.Adam(model.parameters(),lr=.003)
        history=[]
        for epoch in range(args.epochs):
            total=0.0;model.train()
            for xb,yb in loader:
                optimizer.zero_grad(set_to_none=True)
                loss=F.cross_entropy(model(xb),yb)
                loss.backward();optimizer.step();model.project()
                total+=float(loss.detach())*len(yb)
            with torch.no_grad():
                val_accuracy=float((model(x[val_ids]).argmax(1)==y[val_ids]).float().mean())
            history.append({"epoch":epoch+1,"train_ce":total/len(train_ids),"validation_accuracy":val_accuracy})
        with torch.no_grad():
            logits=model(xt).numpy()
        arrays={"weight_0":model.first.weight.detach().numpy().T,
                "weight_1":model.second.weight.detach().numpy().T,
                "bias_0":model.first.bias.detach().numpy(),"bias_1":model.second.bias.detach().numpy(),
                "reference_test_logits":logits}
        np.savez_compressed(out/f"seed{seed}.npz",**arrays)
        metrics.append({"seed":seed,"epochs":args.epochs,"test_examples":len(yt),
                        "fp32_accuracy":float(np.mean(logits.argmax(1)==yt.numpy())),
                        "logical_weight_pbits":sum(arrays[k].size for k in ("weight_0","weight_1")),
                        "all_weights_in_unit_interval":bool(all(np.all((arrays[k]>=0)&(arrays[k]<=1)) for k in ("weight_0","weight_1"))),
                        "deterministic_bias_parameters":42,"history":history})
        print({k:v for k,v in metrics[-1].items() if k!="history"},flush=True)
    report={"torch_version":torch.__version__,"seeds":args.seeds,"epochs":args.epochs,
            "train_examples":55000,"validation_examples":5000,"test_examples":10000,
            "architecture":"64->32->10, ReLU, positive projected weights",
            "preprocessing":"28x28 to8x8 adaptive average pooling; (x-0.1307)/0.3081",
            "training_optimizer":"Adam lr0.003; project all matrix weights to [0,1] after every update",
            "seed_before_model_construction":True,"elapsed_seconds":time.monotonic()-start,
            "models":metrics}
    (results/"manifest.json").write_text(json.dumps(report,indent=2)+"\n")


if __name__=="__main__":main()
