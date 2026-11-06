"""Persistent native THRML operator worker for the separate Torch environment."""
import os
os.environ.setdefault('JAX_PLATFORMS','cpu')
import json
import sys
from pathlib import Path
import time
import jax
import jax.numpy as jnp
import numpy as np
from thrml import SamplingSchedule,sample_with_observation
from research.thrml_spw.native import NativeProbabilityBank
from research.thrml_spw.controlled_positive_native import set_fields
from research.thrml_spw.lfm3.native_operator_pilot_v2 import Sum


def main():
    cache={};functions={}
    print(json.dumps({'ready':True,'devices':[str(d) for d in jax.devices()]}),flush=True)
    for line in sys.stdin:
        req=json.loads(line)
        if req.get('stop'):break
        start=time.monotonic();a=np.load(req['input']);xf=a['x'];w=a['w'];conv=req['kind']=='conv'
        scale=float(np.max(np.abs(xf)))/127 if np.any(xf) else 1.0
        x=np.rint(xf/scale).clip(-127,127).astype(np.int32)
        t=int(req['samples']);fanin=w.shape[0];shape=(fanin,64)
        if shape not in cache:cache[shape]=NativeProbabilityBank(np.full(shape,.5))
        bank=cache[shape]
        assert 127*fanin*t < np.iinfo(np.int32).max
        keyfn=(shape,conv,t,x.shape[0])
        if keyfn not in functions:
            def build(bank,conv,t):
                initial=[jnp.zeros(bank.logical_pbits,dtype=jnp.bool_)]
                def run(key,inputs,fields):
                    program=set_fields(bank,fields);obs=Sum(bank.readout,inputs,bank.shape,conv)
                    total,_=sample_with_observation(key,program,SamplingSchedule(1,t,1),initial,[],obs.init(),obs)
                    return total
                return jax.jit(run)
            functions[keyfn]=build(bank,conv,t)
        run=functions[keyfn];ys=[];padding=0
        for tile,first in enumerate(range(0,w.shape[1],64)):
            width=min(64,w.shape[1]-first)
            wt=w[:,first:first+width].astype(np.float64)
            if width<64:wt=np.pad(wt,((0,0),(0,64-width)));padding+=fanin*(64-width)
            lo=wt.min(0);span=np.ptp(wt,axis=0)
            p=np.divide(wt-lo,span,out=np.full_like(wt,.5),where=span!=0).clip(1e-12,1-1e-12)
            h=np.clip((np.log(p)-np.log1p(-p))/2,-4,4)
            codes=np.rint((h+4)*255/8).clip(0,255).astype(np.uint8)
            fields=(-4+codes.astype(np.float32)*np.float32(8/255)).reshape(-1)
            xt=x[:,:,first:first+width] if conv else x
            if conv and width<64:xt=np.pad(xt,((0,0),(0,0),(0,64-width)))
            offset=xt.sum(1) if conv else xt.sum(-1,keepdims=True)
            if req['mode']=='native':
                key=jax.random.fold_in(jax.random.fold_in(jax.random.key(req['key']),req['call']),tile)
                total=np.asarray(run(key,jnp.asarray(xt),jnp.asarray(fields)))
                y=(total.astype(np.float64)/t*span+offset*lo)*scale
            else:
                prob=1/(1+np.exp(-2*fields.astype(np.float64))).reshape(wt.shape)
                wm=lo+span*prob
                y=np.einsum('bki,ki->bi',xt.astype(np.float64)*scale,wm) if conv else (xt.astype(np.float64)*scale)@wm
            ys.append(y[:,:width])
        result=np.concatenate(ys,axis=1);np.save(req['output'],result)
        print(json.dumps({'ok':True,'call':req['call'],'weight_coordinates':w.size,
            'padding_nodes':padding,'retained_evaluations':int((w.size+padding)*t) if req['mode']=='native' else 0,
            'warmup_evaluations':int(w.size+padding) if req['mode']=='native' else 0,
            'elapsed_seconds':time.monotonic()-start,'finite':bool(np.isfinite(result).all())}),flush=True)


if __name__=='__main__':main()
