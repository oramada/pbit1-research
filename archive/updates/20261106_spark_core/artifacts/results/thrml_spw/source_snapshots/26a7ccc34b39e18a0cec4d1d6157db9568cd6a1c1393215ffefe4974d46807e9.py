"""Standalone scientific figures from completed native-simulator evidence."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def rows(path):
    with Path(path).open() as f:return list(csv.DictReader(f))


def main():
    root=Path("results/thrml_spw")
    scalar=rows(root/"scalar_full/sampling_metrics.csv")
    means=rows(root/"scalar_full/weight_probabilities.csv")
    neural=rows(root/"positive_full_test/summary.csv")
    fig,axes=plt.subplots(1,3,figsize=(14,4),layout="constrained")
    axes[0].plot([float(r["weight"]) for r in means],[float(r["measured_p"]) for r in means],color="tab:blue",label="Native THRML")
    axes[0].plot([0,1],[0,1],"k--",lw=1,label="p=w")
    axes[0].set(xlabel="Target weight w",ylabel="Measured P(z=1)",title="One logical p-bit per weight")
    axes[0].legend()
    t=[int(r["samples"]) for r in scalar]
    axes[1].loglog(t,[float(r["rmse"]) for r in scalar],"o-",label="Native THRML")
    axes[1].loglog(t,[float(r["theory_rmse"]) for r in scalar],"k--",label="Bernoulli prediction")
    axes[1].set(xlabel="Samples per weight",ylabel="Reconstruction RMSE",title="1,001 weights; 1,000 trials each")
    axes[1].legend()
    native=[r for r in neural if r["representation"]=="THRML_layer_preactivation"]
    axes[2].errorbar([int(r["samples_per_layer"]) for r in native],
        [100*float(r["accuracy_mean"]) for r in native],
        yerr=[100*float(r["accuracy_ci95_margin"]) for r in native],fmt="o-",label="Native THRML")
    for rep,color,label in (("FP32","black","Same model FP32"),
                            ("UINT4_per_layer_scale","tab:orange","Same model UINT4")):
        r=next(r for r in neural if r["representation"]==rep)
        axes[2].axhline(100*float(r["accuracy_mean"]),color=color,linestyle="--",label=label)
    axes[2].set(xscale="log",xlabel="Samples per layer",ylabel="Test accuracy (%)",
                title="Learned w=p network; full MNIST test")
    axes[2].legend(fontsize=8)
    fig.savefig(root/"native_evidence.png",dpi=200)
    fig.savefig(root/"native_evidence.pdf")
    print(root/"native_evidence.png")


if __name__=="__main__":main()
