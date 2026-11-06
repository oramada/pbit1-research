"""Native grouped output accumulation, all parameter kinds, finite controls."""
import os
os.environ.setdefault('JAX_PLATFORMS','cpu')
import json
from pathlib import Path
import sys
import time
import jax
import jax.numpy as jnp
import numpy as np
from thrml import (AbstractObserver,SamplingSchedule,sample_states,sample_with_observation,
                   block_state_to_global,from_global_state)
from research.thrml_spw.native import NativeProbabilityBank
from research.thrml_spw.controlled_positive_native import set_fields


class GroupSum(AbstractObserver):
    block: object
    x: jax.Array
    shape: tuple
    depthwise: bool
    group_size: int

    def init(self):return jnp.zeros((self.x.shape[0],self.shape[0]//self.group_size,self.shape[1]),jnp.int32)

    def __call__(self,program,free,clamped,carry,iteration):
        state=block_state_to_global(free+clamped,program.gibbs_spec)
        z=from_global_state(state,program.gibbs_spec,[self.block])[0].reshape(-1,self.group_size,self.shape[1]).astype(jnp.int32)
        if self.depthwise:
            x=self.x.reshape(self.x.shape[0],-1,self.group_size,self.shape[1])
            y=jnp.sum(x*z[None,:,:,:],axis=2,dtype=jnp.int32)
        else:y=jnp.einsum('bgk,gko->bgo',self.x.reshape(self.x.shape[0],-1,self.group_size),z)
        return carry+y,None


def main():
    banks={};functions={};public_functions={}
    print(json.dumps({'ready':True,'devices':[str(d) for d in jax.devices()]}),flush=True)
    for line in sys.stdin:
        req=json.loads(line)
        if req.get('stop'):break
        start=time.monotonic();a=np.load(req['input']);xf=a['x'];w=a['w'];kind=req['kind'];depthwise=kind!='linear'
        scale=float(np.abs(xf).max())/127 if np.any(xf) else 1.0
        x=np.rint(xf/scale).clip(-127,127).astype(np.int32)
        t=int(req['samples']);fanin=w.shape[0];size=min(128,fanin);shape=(fanin,64)
        assert fanin%size==0 and 127*size*t < np.iinfo(np.int32).max
        groups=fanin//size
        if shape not in banks:banks[shape]=NativeProbabilityBank(np.full(shape,.5))
        bank=banks[shape];cachekey=(shape,depthwise,t,x.shape[0])
        if cachekey not in functions:
            def build(bank,size,t,depthwise):
                initial=[jnp.zeros(bank.logical_pbits,dtype=jnp.bool_)]
                def forward(key,inputs,fields):
                    obs=GroupSum(bank.readout,inputs,bank.shape,depthwise,size)
                    total,_=sample_with_observation(key,set_fields(bank,fields),SamplingSchedule(1,t,1),initial,[],obs.init(),obs)
                    return total
                def public(key,fields):return sample_states(key,set_fields(bank,fields),SamplingSchedule(1,t,1),initial,[],[bank.readout])[0]
                return jax.jit(forward),jax.jit(public)
            functions[cachekey],public_functions[cachekey]=build(bank,size,t,depthwise)
        run=functions[cachekey];public=public_functions[cachekey];ys=[];padding=0
        for tile,first in enumerate(range(0,w.shape[1],64)):
            width=min(64,w.shape[1]-first);wt=w[:,first:first+width].astype(np.float64)
            if width<64:wt=np.pad(wt,((0,0),(0,64-width)));padding+=fanin*(64-width)
            wg=wt.reshape(groups,size,64)
            lo=wg.min(1);span=np.ptp(wg,axis=1)
            if kind in ['embedding','elementwise']:
                assert fanin==1
                low=np.asarray(a['lower'][first:first+width],dtype=np.float64)
                sp=np.asarray(a['span'][first:first+width],dtype=np.float64)
                if width<64:low=np.pad(low,(0,64-width));sp=np.pad(sp,(0,64-width))
                lo=low[None,:];span=sp[None,:]
            p=np.divide(wg-lo[:,None,:],span[:,None,:],out=np.full_like(wg,.5),where=span[:,None,:]!=0).clip(1e-12,1-1e-12)
            field=np.clip((np.log(p)-np.log1p(-p))/2,-4,4)
            codes=np.rint((field+4)*255/8).clip(0,255).astype(np.uint8)
            fields=(-4+codes.astype(np.float32)*np.float32(8/255)).reshape(-1)
            xt=x[:,:,first:first+width] if depthwise else x
            if depthwise and width<64:xt=np.pad(xt,((0,0),(0,0),(0,64-width)))
            xg=xt.reshape(xt.shape[0],groups,size,64) if depthwise else xt.reshape(xt.shape[0],groups,size)
            offset=xg.sum(2) if depthwise else xg.sum(2)[:,:,None]
            if req['mode']=='native':
                key=jax.random.fold_in(jax.random.fold_in(jax.random.key(req['key']),req['call']),tile)
                total=np.asarray(run(key,jnp.asarray(xt),jnp.asarray(fields)))
                if req.get('replay',False):
                    states=np.asarray(public(key,jnp.asarray(fields))).reshape(t,groups,size,64)
                    check=np.zeros_like(total,dtype=np.int64)
                    for state in states:
                        if depthwise:check+=np.sum(xg.astype(np.int64)*state[None,:,:,:],axis=2,dtype=np.int64)
                        else:
                            for group in range(groups):check[:,group,:]+=xg[:,group,:].astype(np.int64)@state[group].astype(np.int64)
                    np.testing.assert_array_equal(total.astype(np.int64),check)
                    if req.get('state_dir'):
                        path=Path(req['state_dir']);path.mkdir(exist_ok=True)
                        np.savez_compressed(path/f"call{req['call']}_tile{tile}.npz",states=states,codes=codes,total=total,inputs=xt)
                y=np.sum(total.astype(np.float64)/t*span[None,:,:]+offset*lo[None,:,:],axis=1)*scale
            else:
                q=1/(1+np.exp(-2*fields.astype(np.float64))).reshape(wg.shape)
                wm=lo[:,None,:]+span[:,None,:]*q
                y=np.sum(xg.astype(np.float64)*wm[None,:,:,:],axis=(1,2))*scale if depthwise else np.einsum('bgk,gko->bo',xg.astype(np.float64)*scale,wm)
            ys.append(y[:,:width])
        result=np.concatenate(ys,axis=1);np.save(req['output'],result)
        print(json.dumps({'ok':True,'call':req['call'],'weight_coordinates':w.size,'padding_nodes':padding,
            'public_replay_checked':req.get('replay',False),'retained_evaluations':int((w.size+padding)*t) if req['mode']=='native' else 0,
            'warmup_evaluations':int(w.size+padding) if req['mode']=='native' else 0,
            'elapsed_seconds':time.monotonic()-start,'finite':bool(np.isfinite(result).all())}),flush=True)


if __name__=='__main__':main()
