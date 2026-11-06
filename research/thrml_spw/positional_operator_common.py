"""Finite-field positional/temporal operators: shared frozen controls and metrics."""
from pathlib import Path
import json,hashlib
import numpy as np
OUT=Path('results/thrml_spw/positional_operators')
ROOT=Path('results/thrml_spw/range_localization')
SOURCES=['positional_operator_common.py','prepare_positional_operators.py','positional_operator_native.py','positional_operators.py','verify_positional_operators.py','verify_positional_operator_controls.py','positional_operator_report.py','native.py','controlled_positive_native.py','provenance.py']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def guard(p):
    for f,h in p['source_sha256'].items():assert sha(f)==h,(f,h)
def controls(entry,k):
    with np.load(entry['path']) as a:
        shape=a['codes'].shape;lower=np.repeat(a['lower'],16,0);span=np.repeat(a['span'],16,0);bias=a['bias'].copy();codes=a['codes'].copy()
    if k==1:fields=(-4+codes.astype(float)*8/255).astype(np.float32)[...,None]
    else:fields=np.where((codes[...,None]>>np.arange(4))&1,4,-4).astype(np.float32)
    return shape,lower,span,bias,fields

def prediction(x,target,lower,span,bias,fields,k,t):
    weights=2.**np.arange(k)/((1<<k)-1);q=np.exp(-np.logaddexp(0,-2*fields.astype(float)));mean=np.sum(q*weights,axis=-1);var=np.sum(q*(1-q)*weights**2,axis=-1);ym=x.astype(float)@(lower.astype(float)+span.astype(float)*mean)+bias
    v=float(np.mean((x.astype(float)**2)@(span.astype(float)**2*var))/t);b=float(np.mean((ym-target)**2))
    return ym,{'predicted_sampling_MSE':v,'programmed_mean_bias_MSE':b,'predicted_target_MSE':v+b}
