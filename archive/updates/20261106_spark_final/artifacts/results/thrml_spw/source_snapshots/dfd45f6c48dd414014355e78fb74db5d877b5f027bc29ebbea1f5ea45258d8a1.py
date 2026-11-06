"""Monte Carlo proof of w=p with actual THRML sample trajectories."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import time

import numpy as np
from scipy.stats import binom

from .native import NativeProbabilityBank
from .provenance import manifest

BUDGETS = np.array([1,2,4,8,16,32,64,128,256,512,1024,4096])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", type=int, default=1001)
    parser.add_argument("--trials-per-seed", type=int, default=200)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--batch-trials", type=int, default=50)
    parser.add_argument("--out", default="results/thrml_spw/scalar_full")
    args = parser.parse_args()
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    targets=np.linspace(0,1,args.targets)
    bank=NativeProbabilityBank(targets)
    counts=np.zeros((len(args.seeds),args.trials_per_seed,len(BUDGETS),len(targets)),dtype=np.uint16)
    start=time.monotonic()
    for si,seed in enumerate(args.seeds):
        for begin in range(0,args.trials_per_seed,args.batch_trials):
            n=min(args.batch_trials,args.trials_per_seed-begin)
            # Fixed keys and recorded batching define the exact random trajectories.
            sample_seed=seed*1000003+begin
            bits=bank.sample(sample_seed,int(BUDGETS[-1]),n)
            prefix=np.cumsum(bits,axis=1,dtype=np.uint16)
            counts[si,begin:begin+n]=prefix[:,BUDGETS-1,:]
            if si==0 and begin==0:
                np.savez_compressed(out/"example_trajectories.npz",weights=targets,bits=bits[:2])
            del prefix,bits
            print(f"seed={seed} trials={begin+n}/{args.trials_per_seed} elapsed={time.monotonic()-start:.1f}s",flush=True)
    np.savez_compressed(out/"prefix_counts.npz",counts=counts,weights=targets,budgets=BUDGETS,seeds=args.seeds)
    flat=counts.reshape((-1,len(BUDGETS),len(targets))).astype(np.float64)
    ntrials=len(flat)
    estimates=flat/BUDGETS[None,:,None]
    error=estimates-targets[None,None,:]
    variance=targets*(1-targets)
    aggregate_counts=flat.sum(0).astype(np.int64)
    trials_per_point=ntrials*BUDGETS[:,None]
    equal_tail_p=np.minimum(1,2*np.minimum(binom.cdf(aggregate_counts,trials_per_point,targets),
                                           binom.sf(aggregate_counts-1,trials_per_point,targets)))
    family_alpha=0.01
    threshold=family_alpha/equal_tail_p.size
    rows=[]; mse_z=[]
    for ti,t in enumerate(BUDGETS):
        empirical_mse=float(np.mean(error[:,ti]**2)); theory=float(np.mean(variance)/t)
        square_error_var=2*variance**2/t**2+variance*(1-6*variance)/t**3
        se=float(np.sqrt(np.sum(np.maximum(square_error_var,0))/(ntrials*len(targets)**2)))
        z=(empirical_mse-theory)/se if se else 0
        mse_z.append(float(z))
        rows.append({"samples":int(t),"trials_per_target":ntrials,
                     "targets":len(targets),"bias":float(np.mean(error[:,ti])),
                     "mae":float(np.mean(np.abs(error[:,ti]))),
                     "rmse":float(np.sqrt(empirical_mse)),"theory_rmse":float(np.sqrt(theory)),
                     "mse_ratio":empirical_mse/theory,"mse_z":float(z),
                     "worst_case_std":float(.5/np.sqrt(t))})
    with (out/"sampling_metrics.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    with (out/"weight_probabilities.csv").open("w",newline="") as f:
        fields=["weight","programmed_bias","control_realized_p","measured_p","binomial_pvalue_T4096"]
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        biases=np.full(len(targets),np.nan);biases[bank.free_indices]=bank.biases
        for i,w in enumerate(targets):
            writer.writerow(dict(zip(fields,[float(w),float(biases[i]),
                float(bank.realized_probabilities[i]),float(np.mean(estimates[:,-1,i])),float(equal_tail_p[-1,i])])))
    endpoints_exact=bool(np.all(counts[:,:,:,0]==0) and
                         np.all(counts[:,:,:,-1]==BUDGETS[None,None,:]))
    result={"native_backend":bank.controls(),"total_trials_per_target":ntrials,
            "all_endpoint_outputs_exact":endpoints_exact,
            "maximum_control_rounding_error":float(np.max(np.abs(bank.realized_probabilities-targets))),
            "max_abs_mse_z":max(abs(z) for z in mse_z),
            "bonferroni_family_alpha":family_alpha,"per_test_threshold":threshold,
            "binomial_rejected_conditions":int(np.sum(equal_tail_p<threshold)),
            "minimum_binomial_pvalue":float(equal_tail_p.min()),
            "elapsed_seconds":time.monotonic()-start}
    result["passed"]=(endpoints_exact and result["max_abs_mse_z"]<6 and
                       result["binomial_rejected_conditions"]==0)
    manifest(out/"manifest.json",vars(args)|{"sample_counts":BUDGETS.tolist(),
             "warmup":1,"steps_per_sample":1,"initial_state":False},result)
    # Use standard scientific plotting tools, retaining standalone PDF/PNG artifacts.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout="constrained")
    axes[0].plot(targets,np.mean(estimates[:,-1],axis=0),label="THRML measurements")
    axes[0].plot([0,1],[0,1],"k--",label="p=w")
    axes[0].set(xlabel="Encoded weight w",ylabel="Measured P(z=1)",title="One logical p-bit per weight")
    axes[0].legend()
    axes[1].loglog(BUDGETS,[row["rmse"] for row in rows],"o-",label="THRML RMSE")
    axes[1].loglog(BUDGETS,[row["theory_rmse"] for row in rows],"k--",label="Bernoulli theory")
    axes[1].set(xlabel="Samples per weight",ylabel="Reconstruction RMSE",title=f"{ntrials} trials per target")
    axes[1].legend()
    fig.savefig(out/"scalar_proof.png",dpi=180);fig.savefig(out/"scalar_proof.pdf")
    plt.close(fig)
    print(result,flush=True)
    if not result["passed"]:
        raise RuntimeError("Scalar statistical validation failed; inspect manifest before claiming success")


if __name__ == "__main__":main()
