"""Frozen scalar finite-field positional versus temporal comparison definitions."""
from pathlib import Path
import hashlib,json
import numpy as np
OUT=Path('results/thrml_spw/finite_positional')
SOURCES=['finite_positional_common.py','prepare_finite_positional.py','finite_positional.py','verify_finite_positional.py','finite_positional_report.py','native.py','provenance.py']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def design():
    return {'targets':257,'trials_per_seed':200,'seeds':list(range(5)),'chain_batch':50,'field_bounds':[4,2],'spatial_bits':[2,4,8],'parallel_samples':[1,4,16,64],'single_samples':[1,2,4,8,16,32,64,128,256,512],'single_control_bits':8,'key_root':1227500,'warmup':1,'steps_per_sample':1,'initial_state':False,'gate_max_abs_mse_z':6,'scope':'Internal positional binary word in finite-bias official THRML nodes; not an asserted Extropic architecture. Scalar reconstruction only; no neural/hardware benefit claim.'}
def parameters(p,bound,k):
    target=np.linspace(0,1,p['targets']);maximum=(1<<k)-1
    if k==1:
        clipped=np.clip(target,1/(1+np.exp(2*bound)),1/(1+np.exp(-2*bound)))
        field=np.clip(.5*(np.log(clipped)-np.log1p(-clipped)),-bound,bound)
        codes=np.rint((field+bound)*255/(2*bound)).astype(np.uint8)
        fields=(-bound+codes.astype(float)*(2*bound)/255).astype(np.float32)[:,None];coeff=np.ones(1)
    else:
        q=np.rint(target*maximum).astype(np.uint16);bits=((q[:,None]>>np.arange(k))&1).astype(np.uint8)
        fields=np.where(bits,bound,-bound).astype(np.float32);codes=bits;coeff=2.**np.arange(k)/maximum
    prob=np.exp(-np.logaddexp(0,-2*fields.astype(float)))
    return target,fields,codes,coeff,prob
def seed_for(p,bound,k,seed,start):return p['key_root']+1000000*bound+100000*k+1000*seed+start
def source_guard(p):
    for f,h in p['source_sha256'].items():assert sha(f)==h,(f,h)
def metrics(counts,target,prob,coeff,t):
    estimates=np.einsum('srwk,k->srw',counts.astype(float)/t,coeff);errors=estimates-target
    bias=prob@coeff-target;v=np.sum(prob*(1-prob)*coeff**2,axis=1)/t
    k3=np.sum(prob*(1-prob)*(1-2*prob)*coeff**3,axis=1)/t**2
    k4=np.sum(prob*(1-prob)*(1-6*prob*(1-prob))*coeff**4,axis=1)/t**3
    theory=float(np.mean(v+bias**2));mse=float(np.mean(errors**2));trials=counts.shape[0]*counts.shape[1]
    se=float(np.sqrt(np.maximum(4*bias**2*v+4*bias*k3+k4+2*v**2,0).sum()/(trials*len(target)**2)))
    return {'samples':t,'pbits_per_scalar':len(coeff),'node_updates_per_scalar_trial':t*len(coeff),'mean_bias':float(errors.mean()),'mae':float(np.abs(errors).mean()),'mse':mse,'rmse':float(np.sqrt(mse)),'theory_mse':theory,'mse_standard_error':se,'mse_z':(mse-theory)/se,'max_absolute_stationary_bias':float(np.max(np.abs(bias))),'theory_variance':float(v.mean()),'theory_squared_bias':float((bias**2).mean()),'seed_mse':np.mean(errors**2,axis=(1,2)).tolist()}
