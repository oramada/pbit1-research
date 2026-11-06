"""Frozen post-selection full-test evaluation on reserved static device/MC seeds."""
import csv,json,time
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from .probability_training_native import profile,command,initial_theta
from .controlled_positive_native import ControlledPositiveNetwork
from .finite_endpoint_control import quantize,decode
from .gain_calibration_math import fit_two_probe
from .calibrated_mnist import forward,scores,digest
from .verify_transformer_control import sha
from .provenance import manifest
OUT=Path('results/thrml_spw/probability_finetune');DATA=Path('data/thrml_spw/positive_mnist')
KINDS=['initial','expected','native_ideal','native_device'];CONDITIONS=['ideal','uncalibrated','cal4096']


def main():
    if jax.default_backend()!='gpu':raise RuntimeError('CUDA full evaluation required')
    train_report=json.loads((OUT/'TRAINING_REPORT.json').read_text());selection=json.loads((OUT/'selected.json').read_text());assert len(selection)==15 and train_report['selection_sha256']==sha(OUT/'selected.json')
    for folder in ['test_runs','test_devices','test_baselines']:(OUT/folder).mkdir(exist_ok=True)
    protocol={'schema':'spw-probability-finetune-held-device-test-v1','training_protocol_sha256':sha(OUT/'protocol.json'),'training_manifest_sha256':sha(OUT/'training_manifest.json'),
        'selection_sha256':sha(OUT/'selected.json'),'checkpoint_kinds':KINDS,'conditions':CONDITIONS,'models':list(range(5)),'reserved_profiles':list(range(5)),'samples_per_layer':[64,256],'examples':10000,'batch':32,
        'keys':'Test1130730foldmodel,device,T,thendatasetbatchandsplitlayers;sharedacrossmethods/conditions. Probe1130731foldmodel,device,probe0/1;no overlapwithtrain1110730orvalidation1120730.',
        'profiles':'Reserved test-domainprofile(base2026112600) staticgainlognormalmean1/logSD.2/offsetSD.2. Counts2048perprobe at8bitcodes96/159; measuredgain/offsetfit reusedacrossallfourcheckpoints onthe same conceptualdevices. Calibrationneverreceivestrueparams orlabels. No drift.',
        'decision':'All600full-testcases and50probe windows retained. Primarynative_device minusexpected onuncalibratedT64. Otherconditions,budgetsandnativeidealcontrolsecondary. Selectionwas frozenbeforetestinputs/scoreswere loaded; existingtestsetalreadyusedinearlierwork, notfreshdata.',
        'scope':'Five pairedfine-tunedstartingmodels withindependentreserveddeviceandMCseeds, notfifteenindependentinitializations. Entirefreeweightbanks;noendpointclamps;deterministicbiasesfrozen. Finitecontrolsandfloatingexternalarithmetic. Nohardwareadvantageclaim.',
        'source_sha256':{str(Path('research/thrml_spw')/f):sha(Path('research/thrml_spw')/f) for f in ['probability_finetune_evaluate.py','probability_training_native.py','controlled_positive_native.py','finite_endpoint_control.py','gain_calibration_math.py','calibrated_mnist.py']},
        'evaluation_data_sha256':sha(DATA/'evaluation_data.npz')}
    pp=OUT/'test_protocol.json'
    if pp.exists():assert json.loads(pp.read_text())==protocol
    else:pp.write_text(json.dumps(protocol,indent=2)+'\n')
    archive=Path('results/thrml_spw/source_snapshots');archive.mkdir(exist_ok=True)
    for n,h in protocol['source_sha256'].items():
        path=archive/(h+'.py');data=Path(n).read_bytes()
        if path.exists():assert path.read_bytes()==data
        else:path.write_bytes(data)
    d=np.load(DATA/'evaluation_data.npz');x,y=d['test_inputs'],d['test_labels'];xb=jnp.asarray(np.pad(x,((0,16),(0,0))).reshape(313,32,64))
    rows=[];base=[];devices=[];start=time.monotonic()
    for model in range(5):
        e=np.load(DATA/f'seed{model}.npz');ww=[e[f'weight_{i}'] for i in range(2)];bias=[e[f'bias_{i}'] for i in range(2)];reference=forward(x,ww,bias)
        np.testing.assert_allclose(reference,e['reference_test_logits'],atol=1e-4)
        templates=[np.full((64,32),.5),np.full((32,10),.5)];net=ControlledPositiveNetwork(templates,bias);assert len(net.calibration_bank.free_indices)==2368 and not len(net.calibration_bank.clamped_indices)
        fns={t:net.dataset(t) for t in [64,256]};counter=net.counts(2048)
        checkpoints={'initial':initial_theta(ww)}
        for kind in KINDS[1:]:
            cp=OUT/'selected'/f'model{model}_{kind}.npz';assert sha(cp)==selection[f'model{model}_{kind}']['checkpoint_sha256'];checkpoints[kind]=np.load(cp)['theta']
        for kind,theta in checkpoints.items():
            p=(1/(1+np.exp(-theta.astype(float)))).astype(np.float32);w=[p[:2048].reshape(64,32),p[2048:].reshape(32,10)];arr={'original_FP32':reference,'expected':forward(x,w,bias)}
            for bits in [4,8]:
                q=[]
                for a in w:
                    scale=float(a.max())/((1<<bits)-1);q.append((np.rint(a/scale)*scale).astype(np.float32))
                arr[f'UINT{bits}']=forward(x,q,bias)
            np.savez_compressed(OUT/'test_baselines'/f'model{model}_{kind}.npz',**arr)
            for condition,v in arr.items():base.append({'model':model,'kind':kind,'condition':condition,**scores(v,y,reference)})
        probe_fields=[float(decode(np.array(c,np.uint8),8,4)) for c in [96,159]]
        for device in range(5):
            cp=OUT/'test_devices'/f'model{model}_device{device}.npz';cj=cp.with_suffix('.json');gain,offset=profile(model,device,'test')
            if cj.exists():
                dm=json.loads(cj.read_text());assert sha(cp)==dm['archive_sha256'];cal=dict(np.load(cp))
            else:
                counts=[]
                for pi,pf in enumerate(probe_fields):
                    fields=(gain*pf+offset).astype(np.float32);key=jax.random.key(1130731)
                    for v in [model,device,pi]:key=jax.random.fold_in(key,v)
                    counts.append(np.asarray(counter(key,jnp.asarray(fields))))
                ah,dh,raw=fit_two_probe(*counts,2048,*probe_fields)
                cal={'gain':gain,'offset':offset,'count0':counts[0],'count1':counts[1],'estimated_gain':ah,'estimated_offset':dh}
                np.savez_compressed(cp,**cal);dm={'model':model,'device':device,'archive_sha256':sha(cp),'gain_guard_low':int(np.count_nonzero(raw<.25)),'gain_guard_high':int(np.count_nonzero(raw>4)),'native_calibration_free_evaluations':4096*2368}
                cj.write_text(json.dumps(dm,indent=2)+'\n')
            devices.append(dm)
            for kind,theta in checkpoints.items():
                nominal=command(theta);corrected=decode(quantize((nominal.astype(float)-cal['estimated_offset'])/cal['estimated_gain'],8,4),8,4)
                fields_by_condition={'ideal':nominal,'uncalibrated':(gain*nominal.astype(float)+offset).astype(np.float32),'cal4096':(gain*corrected.astype(float)+offset).astype(np.float32)}
                for t in [64,256]:
                    key=jax.random.key(1130730)
                    for v in [model,device,t]:key=jax.random.fold_in(key,v)
                    for condition,fields in fields_by_condition.items():
                        name=f'model{model}_device{device}_{kind}_{condition}_T{t}';rn=OUT/'test_runs'/f'{name}.npz';rj=rn.with_suffix('.json')
                        if rj.exists():
                            row=json.loads(rj.read_text());assert sha(rn)==row['archive_sha256'] and row['fields_sha256']==digest(fields)
                        else:
                            pre,logits=[np.asarray(v).reshape(-1,v.shape[-1])[:10000] for v in fns[t](key,xb,jnp.asarray(fields))]
                            np.savez_compressed(rn,preactivation0=pre,logits=logits)
                            row={'model':model,'device':device,'kind':kind,'condition':condition,'T':t,**scores(logits,y,reference),'archive_sha256':sha(rn),'fields_sha256':digest(fields),'inference_free_evaluations':313*t*2368,'logical_weight_readouts':313*t*2368,'binary_products_with_padding':10016*t*2368,'binary_products_without_padding':10000*t*2368}
                            rj.write_text(json.dumps(row,indent=2)+'\n')
                        rows.append(row)
            print(f'Reserved test model{model} device{device}, elapsed={time.monotonic()-start:.1f}s',flush=True)
        jax.clear_caches()
    assert len(rows)==600 and len(base)==80 and len(devices)==25 and sha(OUT/'selected.json')==protocol['selection_sha256']
    for name,data in [('test_metrics.csv',rows),('test_baseline_metrics.csv',base)]:
        with (OUT/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    result={'native_full_test_runs':600,'native_probe_windows':50,'native_test_free_evaluations':sum(r['inference_free_evaluations'] for r in rows),'native_test_calibration_free_evaluations':sum(r['native_calibration_free_evaluations'] for r in devices),'selection_frozen_before_test':protocol['selection_sha256'],'elapsed_seconds':time.monotonic()-start,'scope':protocol['scope']}
    (OUT/'TEST_REPORT.json').write_text(json.dumps(result,indent=2)+'\n');manifest(OUT/'test_manifest.json',protocol,result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
