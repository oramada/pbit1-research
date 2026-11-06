"""Predeclared crossed seed/bank resampling for paired confirmation losses.

This is a sensitivity interval under exchangeable clusters, not a finite-sample
coverage guarantee. Bank clusters preserve the eight contexts sharing a weight draw.
"""
import numpy as np


def bootstrap_paired_difference(left,right,draws=10000,seed=20261106,bank_size=8):
    left,right=np.asarray(left,dtype=np.float64),np.asarray(right,dtype=np.float64)
    if left.ndim!=2 or left.shape!=right.shape or left.shape[1]%bank_size:
        raise ValueError('Matched seed-by-window arrays and full bank clusters required')
    seeds,windows=left.shape;banks=windows//bank_size
    delta=(left-right).reshape(seeds,banks,bank_size).mean(-1)
    rng=np.random.default_rng(seed)
    si=rng.integers(0,seeds,size=(draws,seeds));bi=rng.integers(0,banks,size=(draws,banks))
    sampled=delta[si[:,:,None],bi[:,None,:]].mean(axis=(1,2))
    return sampled
