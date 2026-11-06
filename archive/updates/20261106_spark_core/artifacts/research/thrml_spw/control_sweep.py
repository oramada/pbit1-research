"""Native THRML scalar sweep over finite bias precision and dynamic range."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import time

import numpy as np

from .control_bank import ControlledProbabilityBank
from .native import NativeProbabilityBank
from .provenance import manifest


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--seeds",type=int,nargs="+",default=list(range(5)))
    parser.add_argument("--chains",type=int,default=32)
    parser.add_argument("--out",default="results/thrml_spw/control_sweep")
    args=parser.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    weights=np.unique(np.concatenate([np.linspace(0,1,1001),[1e-8,1e-6,1e-4,1-1e-4,1-1e-6,1-1e-8,.73]]))
    budgets=np.array([1,16,64,256,4096])
    settings=[(32,None)]+[(bits,bound) for bound in (2.,4.,8.) for bits in (4,6,8,10,12)]
    all_rows=[];control_rows=[];raw={"weights":weights,"budgets":budgets,"seeds":args.seeds}
    start=time.monotonic();max_z=0
    for bits,bound in settings:
        name="float32" if bound is None else f"b{bits}_r{bound:g}"
        bank=NativeProbabilityBank(weights) if bound is None else ControlledProbabilityBank(weights,bits,bound)
        expected=bank.realized_probabilities
        delta=expected-weights;v=expected*(1-expected)
        counts=[]
        for seed in args.seeds:
            sample=bank.sample(seed,4096,args.chains)
            cumulative=np.cumsum(sample,axis=1,dtype=np.uint16)
            counts.append(cumulative[:,budgets-1,:])
            del sample,cumulative
        counts=np.stack(counts)
        raw[name]=counts
        trials=counts.reshape((-1,len(budgets),len(weights))).astype(np.float64)
        estimates=trials/budgets[None,:,None]
        errors=estimates-weights[None,None,:]
        for ti,t in enumerate(budgets):
            mse=float(np.mean(errors[:,ti]**2))
            predicted=float(np.mean(delta**2+v/t))
            fourth_var=(2*v*v/t**2+v*(1-6*v)/t**3+4*delta*delta*v/t+
                        4*delta*v*(1-2*expected)/t**2)
            mse_se=float(np.sqrt(np.sum(np.maximum(fourth_var,0))/(len(trials)*len(weights)**2)))
            z=(mse-predicted)/mse_se if mse_se else 0
            max_z=max(max_z,abs(z))
            all_rows.append({"condition":name,"field_bits":bits,"field_bound":bound,
                "samples":int(t),"trials":len(trials),"targets":len(weights),
                "empirical_mse":mse,"theoretical_mse_including_control_bias":predicted,
                "mse_z":z,"rmse":float(np.sqrt(mse)),
                "control_only_rmse":float(np.sqrt(np.mean(delta**2))),
                "max_absolute_control_error":float(np.max(np.abs(delta))),
                "sampling_only_rmse":float(np.sqrt(np.mean(v/t)))})
        info=bank.controls()
        control_rows.append({"condition":name,"controls":info,
                             "control_only_rmse":float(np.sqrt(np.mean(delta**2))),
                             "p073_realized":float(expected[np.argmin(np.abs(weights-.73))])})
        print(name,"control RMSE",float(np.sqrt(np.mean(delta**2))),
              "elapsed",round(time.monotonic()-start,1),flush=True)
    np.savez_compressed(out/"prefix_counts.npz",**raw)
    with (out/"metrics.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(all_rows[0]));writer.writeheader();writer.writerows(all_rows)
    result={"max_abs_mse_z":max_z,"validation_threshold":6.,"passed":max_z<6,
            "elapsed_seconds":time.monotonic()-start,"programmed_controls":control_rows,
            "interpretation":"Control-only bias and sampling variance are distinct. Bias does not vanish with T.",
            "physical_scope":"Quantized dimensionless Ising fields, not measured DAC/device performance."}
    manifest(out/"manifest.json",vars(args)|{"budgets":budgets.tolist(),"settings":settings},result)
    if not result["passed"]:raise RuntimeError("Finite-control scalar validation failed")


if __name__=="__main__":main()
