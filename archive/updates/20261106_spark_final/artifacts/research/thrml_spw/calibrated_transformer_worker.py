"""Persistent native all-free Transformer calibration and operator execution."""
import json,sys,time,traceback
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from .calibrated_transformer_native import CalibratedGroupedLinear
from .calibrated_transformer_controls import device,decode
from .finite_endpoint_control import quantize
from .gain_calibration_math import fit_two_probe
from .block_cached_control_worker import sha,atomic_npy,atomic_json
from .calibrated_mnist import digest
from .provenance import manifest


class Worker:
    def __init__(self):
        self.banks={};self.forward_fns={};self.count_fns={};self.source={str(Path('research/thrml_spw')/n):sha(Path('research/thrml_spw')/n) for n in ['calibrated_transformer_worker.py','calibrated_transformer_native.py','calibrated_transformer_controls.py','native.py','grouped_range_native.py','controlled_positive_native.py','calibration_count_native.py','gain_calibration_math.py','finite_endpoint_control.py']}

    def execute(self,r):
        for n,h in self.source.items():assert sha(n)==h
        out=Path(r['output']);meta=out.with_suffix('.json');cp=Path(r['controls']);stage=int(r['stage']);seed=int(r['seed']);condition=r['condition'];assert condition in ['ideal','uncalibrated','cal4096']
        signature={k:r[k] for k in ['controls','stage','seed','condition','samples','contexts_per_bank']};signature.update({'controls_sha256':sha(cp),'input_sha256':sha(r['inputs']),'reference_sha256':sha(r['reference']),'source_sha256':self.source})
        if meta.exists() and out.exists():
            old=json.loads(meta.read_text())
            if old['signature']==signature and old['output_sha256']==sha(out):return {'status':'ok','metadata':str(meta),'cached':True}
        bank_key=(stage,sha(cp))
        if bank_key not in self.banks:
            self.banks[bank_key]=CalibratedGroupedLinear(cp)
            self.forward_fns[bank_key]=self.banks[bank_key].forward(1024)
            self.count_fns[bank_key]=self.banks[bank_key].counts(2048)
        layer=self.banks[bank_key];size=layer.bank.logical_pbits;gain,offset=device(stage,seed,size)
        caldir=Path(r['calibration_folder']);caldir.mkdir(parents=True,exist_ok=True);calp=caldir/f'stage{stage}_seed{seed}.npz';calj=calp.with_suffix('.json')
        if calj.exists():
            cm=json.loads(calj.read_text());assert cm['archive_sha256']==sha(calp) and cm['controls_sha256']==sha(cp) and cm['gain_sha256']==digest(gain) and cm['offset_sha256']==digest(offset);cal=np.load(calp);corrected=cal['corrected_codes']
        else:
            cc=[];ph=[]
            for pi,code in enumerate([96,159]):
                fields=(gain*float(decode(np.array(code,np.uint8)))+offset).astype(np.float32)
                key=jax.random.key(1140731)
                for v in [seed,stage,pi]:key=jax.random.fold_in(key,v)
                cc.append(np.asarray(self.count_fns[bank_key](key,jnp.asarray(fields))));ph.append(digest(fields))
            pf=decode(np.array([96,159],np.uint8)).astype(float);ah,dh,raw=fit_two_probe(cc[0],cc[1],2048,*pf)
            requested=(layer.nominal.astype(float)-dh)/ah;corrected=quantize(requested,8,4)
            np.savez_compressed(calp,count0=cc[0].astype(np.uint16),count1=cc[1].astype(np.uint16),corrected_codes=corrected)
            cm={'stage':stage,'seed':seed,'archive_sha256':sha(calp),'controls_sha256':sha(cp),'gain_sha256':digest(gain),'offset_sha256':digest(offset),'probe_field_sha256':ph,
                'free_nodes':size,'clamped_nodes':0,'probe_runs':2,'free_evaluations':4096*size,'gain_guard_low':int(np.count_nonzero(raw<.25)),'gain_guard_high':int(np.count_nonzero(raw>4)),'commands_bound_clipped':int(np.count_nonzero(np.abs(requested)>4))}
            atomic_json(calj,cm)
        command=layer.nominal if condition!='cal4096' else decode(corrected)
        fields=command if condition=='ideal' else (gain*command.astype(float)+offset).astype(np.float32)
        x,ref=np.load(r['inputs']),np.load(r['reference']);batch=int(r['contexts_per_bank']);t=int(r['samples']);fn=self.forward_fns[bank_key];assert t==1024;ys=[];start=time.monotonic()
        for bank,first in enumerate(range(0,len(x),batch)):
            key=jax.random.key(1140730)
            for v in [seed,stage,bank]:key=jax.random.fold_in(key,v)
            xx=x[first:first+batch];y=np.asarray(fn(key,jnp.asarray(xx.reshape(-1,xx.shape[-1])),jnp.asarray(fields))).reshape(xx.shape[0],xx.shape[1],-1);ys.append(y)
        y=np.concatenate(ys);assert y.shape==ref.shape
        prob=(1/(1+np.exp(-2*fields.astype(float)))).reshape(layer.shape);mean=x.astype(float)@(layer.lower+layer.span*prob)+np.asarray(layer.bias)
        energy=np.mean(x.astype(float)**2,axis=(0,1));variance=float(np.mean(energy@(layer.span.astype(float)**2*prob*(1-prob)))/t)
        atomic_npy(out,y);result={'status':'complete','signature':signature,'output_sha256':sha(out),'calibration_metadata':str(calj),'calibration_archive_sha256':sha(calp),'field_sha256':digest(fields),'free_nodes':size,'clamped_nodes':0,
            'input_shape':list(x.shape),'output_shape':list(y.shape),'bank_batches':len(ys),'free_node_updates':len(ys)*t*size,'logical_weight_readouts':len(ys)*t*size,'binary_product_terms':int(x.shape[0]*x.shape[1]*t*size),
            'conditional_linear_mse':float(np.mean((y.astype(float)-ref)**2)),'conditional_control_bias_mse':float(np.mean((mean-ref)**2)),'predicted_sampling_mse':variance,'sampling_mse_vs_programmed_expectation':float(np.mean((y.astype(float)-mean)**2)),
            'metadata_range_bytes':layer.group_lower.nbytes+layer.group_span.nbytes,'actual_nominal_command_bytes':layer.codes.nbytes,'deterministic_bias_values':int(layer.bias.size),'elapsed_seconds':time.monotonic()-start,
            'scope':'Official THRML all-free normalized-weight operator, static synthetic devices, measured two-probe finite-control compensation. External float arithmetic; no physical performance or direct raw-signed p=w claim.'}
        manifest(out.with_suffix('.manifest.json'),r,result);atomic_json(meta,result)
        return {'status':'ok','metadata':str(meta),'cached':False}


def main():
    if jax.default_backend()!='gpu':raise RuntimeError('CUDA required')
    worker=Worker();print(json.dumps({'status':'ready'}),flush=True)
    for line in sys.stdin:
        try:
            r=json.loads(line)
            if r.get('action')=='quit':break
            print(json.dumps(worker.execute(r)),flush=True)
        except Exception as e:
            traceback.print_exc(file=sys.stderr);print(json.dumps({'status':'error','error':str(e)}),flush=True)


if __name__=='__main__':main()
