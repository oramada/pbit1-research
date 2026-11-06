"""Paired five-model-draw and data-cluster quality uncertainty.

Inputs are paired metric sums within each data unit: complete articles for
language modeling, individual reviews for classification. Positive means worse.
Bootstrap resamples seeds and units independently and retains unit-size weights.
"""
import math
import numpy as np


def summarize(gap_sums,unit_sizes,margin,replicates,key,bootstrap_batch=64):
    gaps=np.asarray(gap_sums,np.float64);sizes=np.asarray(unit_sizes,np.int64)
    if gaps.ndim!=2 or gaps.shape[0]!=5 or sizes.shape!=(gaps.shape[1],):
        raise ValueError('Five paired MC draws and one size per data unit required')
    if not np.isfinite(gaps).all() or not np.all(sizes>0) or not math.isfinite(margin) or margin<0:
        raise ValueError('Finite metric sums, positive cluster sizes and nonnegative margin required')
    if type(replicates)is not int or replicates<100 or type(bootstrap_batch)is not int or bootstrap_batch<1:
        raise ValueError('Valid bootstrap budget and bounded batch required')
    means=gaps.sum(1)/int(sizes.sum());mean=float(means.mean())
    se=float(means.std(ddof=1)/math.sqrt(5));critical=2.7764451051977987
    mc=[mean-critical*se,mean+critical*se]
    rng=np.random.default_rng(key)
    seeds=rng.integers(0,5,(replicates,5))
    draws=np.empty(replicates,np.float64);units=gaps.shape[1]
    for begin in range(0,replicates,bootstrap_batch):
        end=min(begin+bootstrap_batch,replicates);n=end-begin
        chosen=rng.integers(0,units,(n,units))
        for row,selected in enumerate(chosen):
            paired=gaps[seeds[begin+row]][:,selected]
            draws[begin+row]=paired.sum()/(5*int(sizes[selected].sum()))
    boot=np.quantile(draws,[.025,.975],method='linear').tolist()
    return {'per_seed_mean_gap':means.tolist(),'mean_gap':mean,'MC_standard_error':se,
        'MC95_t4_interval_conditional_on_data':mc,'crossed_seed_data_bootstrap_percentile95_interval':boot,
        'bootstrap_replicates':replicates,'bootstrap_key':key,'data_units':units,
        'scored_observations_per_seed':int(sizes.sum()),'margin':margin,
        'MC_upper_within_margin':mc[1]<=margin,'bootstrap_upper_within_margin':boot[1]<=margin,
        'quality_gate_met':mc[1]<=margin and boot[1]<=margin,
        'scope':'Paired metric degradation, fiveMCdraws crossed with complete dataunits; articles retain target-countweights. Approximate intervals, no formal simultaneous/population coverage or numericalBF16claim.'}
