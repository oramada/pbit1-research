"""Standalone figures for native control, correlation, and resource studies."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read(path):
    with Path(path).open() as f:return list(csv.DictReader(f))


def main():
    root=Path("results/thrml_spw")
    fields=read(root/"control_sweep/metrics.csv")
    selected=read(root/"control_neural/summary.csv")
    ideal=read(root/"positive_full_test/summary.csv")
    replicas=read(root/"matched_resources/matched_statistics.csv")
    calibrated=read(root/"coupling_calibration/native_seed_metrics.csv")
    fig,axes=plt.subplots(2,2,figsize=(11,8),layout="constrained")
    for bound in (2,4,8):
        group=[r for r in fields if r["field_bound"] and float(r["field_bound"])==bound and int(r["samples"])==4096]
        group.sort(key=lambda r:int(r["field_bits"]))
        axes[0,0].semilogy([int(r["field_bits"]) for r in group],
            [float(r["control_only_rmse"]) for r in group],"o-",label=f"Bias bound +/-{bound}")
    axes[0,0].set(xlabel="Bias control bits",ylabel="Programming RMSE",title="Control precision and clipping floor")
    axes[0,0].legend()
    for rows,label,color in ((selected,"Selected six-bit bias","tab:orange"),
                             ([r for r in ideal if r["representation"]=="THRML_layer_preactivation" and int(r["samples_per_layer"]) in (64,256)],"Float32 bias","tab:blue")):
        axes[0,1].errorbar([int(r["samples_per_layer"]) for r in rows],
            [100*float(r.get("accuracy_mean",0)) for r in rows],
            yerr=[100*float(r["accuracy_ci95_margin"]) for r in rows],fmt="o-",color=color,label=label)
    fp=next(r for r in ideal if r["representation"]=="FP32")
    axes[0,1].axhline(100*float(fp["accuracy_mean"]),ls="--",color="black",label="Deterministic model")
    axes[0,1].set(xscale="log",xlabel="Samples per layer",ylabel="Full-test accuracy (%)",title="Finite-control native inference")
    axes[0,1].legend(fontsize=8)
    for mode,label,marker in (("one_temporal_pbit","One temporal p-bit","o"),
                              ("K_spatial_same_probability_replicas","K parallel p-bits","x")):
        group=[r for r in replicas if r["mode"]==mode]
        axes[1,0].loglog([int(r["observations_per_scalar"]) for r in group],
                         [float(r["rmse_vs_target"]) for r in group],marker,ls="none",label=label)
    theory={int(r["observations_per_scalar"]):float(r["analytic_rmse_including_float32_bias"]) for r in replicas}
    xs=sorted(theory)
    axes[1,0].loglog(xs,[theory[x] for x in xs],"k--",label="Bernoulli prediction")
    axes[1,0].set(xlabel="Matched observations per scalar",ylabel="Reconstruction RMSE",title="Space/time precision at equal reads")
    axes[1,0].legend(fontsize=8)
    for policy,label,offset in (("independent_logit_fields","Uncorrected fields",-.06),
                                 ("moment_matched_fields","Calibrated fields",.06)):
        group=[r for r in calibrated if int(r["n_spins"])==3 and float(r["coupling_J"])==.8 and int(r["spacing_sweeps"])==1 and r["field_policy"]==policy]
        group.sort(key=lambda r:int(r["node_i"]))
        node=np.array([int(r["node_i"])+1 for r in group])
        values=np.array([float(r["native_marginal_mean_over_seeds"]) for r in group])
        low=np.array([float(r["native_marginal_ci95_low"]) for r in group])
        high=np.array([float(r["native_marginal_ci95_high"]) for r in group])
        axes[1,1].errorbar(node+offset,values,yerr=[values-low,high-values],fmt="o",label=label)
    axes[1,1].plot(node,[float(r["target_p"]) for r in group],"k_",ms=14,label="Requested probability")
    axes[1,1].set(xlabel="Node",ylabel="P(z=1)",title="Known coupling J=0.8; three-spin example",xticks=node)
    axes[1,1].legend(fontsize=8)
    fig.savefig(root/"robustness_evidence.png",dpi=200)
    fig.savefig(root/"robustness_evidence.pdf")
    print(root/"robustness_evidence.png")


if __name__=="__main__":main()
