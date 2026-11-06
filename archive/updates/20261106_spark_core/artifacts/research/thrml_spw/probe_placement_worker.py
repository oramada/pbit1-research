"""Resource-matched actual native probes and counts-only calibrated operators."""
import json,sys,traceback,time
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from .calibrated_transformer_native import CalibratedGroupedLinear
from .probe_placement_math import decode
from .gain_calibration_math import fit_two_probe
from .finite_endpoint_control import quantize
from .calibrated_mnist import digest
from .block_cached_control_worker import sha,atomic_json,atomic_npy
from .provenance import manifest
ROOT=Path('results/thrml_spw/probe_placement')


class Worker:
    def __init__(self):
        self.layers={};self.counts={};self.fns={};self.p=json.loads((ROOT/'protocol.json').read_text());self.designs=json.loads((ROOT/'placement_designs.json').read_text())
        assert sha(ROOT/'placement_designs.json')==self.p['placement_designs_sha256']
        self.sources={str(Path('research/thrml_spw')/n):sha(Path('research/thrml_spw')/n) for n in ['probe_placement_worker.py','probe_placement_math.py','calibrated_transformer_native.py','controlled_positive_native.py','gain_calibration_math.py','finite_endpoint_control.py','native.py','grouped_range_native.py']}
        (ROOT/'calibration').mkdir(exist_ok=True)

    def execute(self,r):
        for n,h in self.sources.items():assert sha(n)==h
        method,stage,seed=r['method'],int(r['stage']),int(r['seed']);assert method in ['fixed','weight_fisher'] and r['samples']==1024
        c=self.p['control_artifacts'][stage];assert sha(c['path'])==c['sha256'];cache=stage,c['sha256']
        if cache not in self.layers:
            self.layers[cache]=CalibratedGroupedLinear(c['path']);self.counts[cache]=self.layers[cache].counts(2048);self.fns[cache]=self.layers[cache].forward(1024)
        layer=self.layers[cache];size=layer.bank.logical_pbits;z=np.random.default_rng(np.random.SeedSequence([2026116600,stage,seed])).standard_normal((2,size));gain=np.exp(.2*z[1]-.02);offset=.2*z[0]
        pair=[96,159] if method=='fixed' else [self.designs[stage]['chosen'][k] for k in ['low_code','high_code']];pf=decode(np.array(pair,np.uint8)).astype(float)
        cp=ROOT/'calibration'/f'{method}_stage{stage}_seed{seed}.npz';cj=cp.with_suffix('.json')
        if cj.exists():
            cm=json.loads(cj.read_text());assert sha(cp)==cm['archive_sha256'] and cm['probe_codes']==pair and cm['controls_sha256']==c['sha256'] and cm['source_sha256']==self.sources and cm['gain_sha256']==digest(gain) and cm['offset_sha256']==digest(offset);bundle=np.load(cp);corrected=bundle['corrected_codes']
        else:
            counts=[];probeh=[]
            for pi,field in enumerate(pf):
                actual=(gain*field+offset).astype(np.float32);key=jax.random.key(1160731)
                for value in [seed,stage,pi]:key=jax.random.fold_in(key,value)
                counts.append(np.asarray(self.counts[cache](key,jnp.asarray(actual))));probeh.append(digest(actual))
            ah,dh,raw=fit_two_probe(counts[0],counts[1],2048,*pf);requested=(layer.nominal.astype(float)-dh)/ah;corrected=quantize(requested,8,4)
            real=(gain*decode(corrected).astype(float)+offset).astype(np.float32);q=1/(1+np.exp(-2*real.astype(float)));q0=1/(1+np.exp(-2*layer.nominal.astype(float)))
            np.savez_compressed(cp,count0=counts[0].astype(np.uint16),count1=counts[1].astype(np.uint16),corrected_codes=corrected)
            cm={'method':method,'stage':stage,'seed':seed,'probe_codes':pair,'probe_fields':pf.tolist(),'probe_field_sha256':probeh,'archive_sha256':sha(cp),'controls_sha256':c['sha256'],'placement_designs_sha256':sha(ROOT/'placement_designs.json'),
                'source_sha256':self.sources,'gain_sha256':digest(gain),'offset_sha256':digest(offset),'gain_RMSE':float(np.sqrt(np.mean((ah-gain)**2))),'offset_RMSE':float(np.sqrt(np.mean((dh-offset)**2))),
                'realized_weight_MSE_vs_nominal_program':float(np.mean(layer.span.reshape(-1).astype(float)**2*(q-q0)**2)),'realized_field_MSE_vs_nominal':float(np.mean((real.astype(float)-layer.nominal.astype(float))**2)),
                'gain_guard_low':int(np.count_nonzero(raw<.25)),'gain_guard_high':int(np.count_nonzero(raw>4)),'requested_commands_clipped':int(np.count_nonzero(np.abs(requested)>4)),'free_nodes':size,'clamped_nodes':0,'probe_runs':2,'free_node_evaluations':4096*size}
            atomic_json(cj,cm);print(f'Probed {method} stage{stage} seed{seed}',file=sys.stderr,flush=True)
        fields=(gain*decode(corrected).astype(float)+offset).astype(np.float32)
        xp,yp=Path(r['inputs']),Path(r['output']);assert sha(xp)==r['input_sha256'];jp=yp.with_suffix('.json')
        signature={'method':method,'stage':stage,'seed':seed,'samples':1024,'contexts_per_bank':8,'inputs_sha256':r['input_sha256'],'controls_sha256':c['sha256'],'calibration_archive_sha256':sha(cp),'source_sha256':self.sources}
        if jp.exists() and yp.exists():
            old=json.loads(jp.read_text())
            if old['signature']==signature and sha(yp)==old['output_sha256']:return {'status':'ok','metadata':str(jp),'cached':True}
        x=np.load(xp);ys=[];start=time.monotonic()
        for bank,first in enumerate(range(0,len(x),8)):
            key=jax.random.key(1160730)
            for value in [seed,stage,bank]:key=jax.random.fold_in(key,value)
            xx=x[first:first+8];value=self.fns[cache](key,jnp.asarray(xx.reshape(-1,xx.shape[-1])),jnp.asarray(fields));ys.append(np.asarray(value).reshape(xx.shape[0],xx.shape[1],-1))
        y=np.concatenate(ys);atomic_npy(yp,y);q=(1/(1+np.exp(-2*fields.astype(float)))).reshape(layer.shape);mean=x.astype(float)@(layer.lower+layer.span*q)+np.asarray(layer.bias)
        variance=float(np.mean(np.mean(x.astype(float)**2,axis=(0,1))@(layer.span.astype(float)**2*q*(1-q)))/1024)
        result={'status':'complete','signature':signature,'output_sha256':sha(yp),'field_sha256':digest(fields),'command_sha256':digest(corrected),'input_shape':list(x.shape),'output_shape':list(y.shape),'free_nodes':size,'clamped_nodes':0,
                'bank_batches':len(ys),'free_node_updates':len(ys)*1024*size,'logical_weight_readouts':len(ys)*1024*size,'binary_product_terms':int(np.prod(x.shape[:2])*1024*size),
                'predicted_sampling_mse':variance,'sampling_mse_vs_programmed_expectation':float(np.mean((y.astype(float)-mean)**2)),'elapsed_seconds':time.monotonic()-start,
                'scope':'Actual official THRML calibrated operator on new student inputs/profiles. Input SHA256 preserved; deterministic inputs independently regenerated from pinned model and prior saved outputs. Counts-only fit, fixedP4096/T1024 and finite8bit controls, no oracle in estimator.'}
        manifest(yp.with_suffix('.manifest.json'),r,result);atomic_json(jp,result);return {'status':'ok','metadata':str(jp),'cached':False}


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
