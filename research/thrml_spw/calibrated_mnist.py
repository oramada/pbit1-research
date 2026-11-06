"""Frozen full-test direct-p=w MNIST study with native measured calibration."""
import csv
import hashlib
import json
import time
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import logsumexp
from .controlled_positive_native import ControlledPositiveNetwork
from .calibration_control_math import decode_command
from .gain_calibration_math import fit_two_probe,corrected_codes
from .provenance import manifest
from .verify_transformer_control import sha

OUT=Path('results/thrml_spw/calibrated_mnist')
DATA=Path('data/thrml_spw/positive_mnist')
CONDITIONS=['ideal8','uncalibrated','cal1024','cal4096','oracle']


def digest(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def forward(x,w,b):return np.maximum(x@w[0]+b[0],0)@w[1]+b[1]


def scores(logits,labels,reference):
    return {'accuracy':float(np.mean(logits.argmax(1)==labels)),
        'cross_entropy':float(np.mean(logsumexp(logits.astype(float),axis=1)-logits[np.arange(len(labels)),labels])),
        'logit_rmse':float(np.sqrt(np.mean((logits.astype(float)-reference)**2)))}


def main():
    if jax.default_backend()!='gpu':raise RuntimeError('This pinned full-network study requires CUDA')
    OUT.mkdir(exist_ok=True)
    for folder in ['baselines','devices','runs']:(OUT/folder).mkdir(exist_ok=True)
    protocol={'schema':'spw-native-calibrated-direct-mnist-v1','model_seeds':list(range(5)),
        'device_sampling_seeds':list(range(10)),'samples_per_layer':[64,256,1024],
        'conditions':CONDITIONS,'calibration_samples_per_free_device':[1024,4096],
        'batch_size':32,'test_examples':10000,'mapping':'Direct p=w for every learned matrix weight; no signed or local-range affine encoding. Exact endpoints stay clamped; neural biases remain deterministic.',
        'control':'Nominal bias float32(logit(w)/2) rounded/clipped to nearest8-bit code in[-4,4]; both measured fits and oracle emit the same finite codes.',
        'device_model':'Static a=exp(.2*z-.02), offset d~N(0,.2^2), realized field=float32(a*b_command+d). Ideal8 uses a=1,d=0. Synthetic dimensionless distributions, not measured device specifications. No drift in this study.',
        'device_rng':'default_rng(2026110100+100*model_seed+device_seed).standard_normal((2,n_free)); row0*.2 is offset, exp(.2*row1-.02) is gain. Same devices across conditions/T/batches.',
        'calibration':'Actual THRML counts for all free model nodes, P/2 samples each at codes96/159; Jeffreys logit line fit, gain guard[.25,4], bounded8-bit inverse commands. No true parameters or labels enter measured fit. One joint bank represents both layers during the probe.',
        'execution':'Native matrix-state output accumulation per layer, then ReLU once. vmap independent dataset batch chains under shared static fields; this is simulation parallelism, not extra physical devices or hardware throughput.',
        'keys':'Inference key1080730 fold model_seed,device_seed,T then batch; split into two layer keys. Identical inference keys across conditions for common random numbers, not common bit states. Probe key1080731 fold model_seed,device_seed,P,probe_code.',
        'padding':'Pad test inputs with zero rows to10016, reshape313 batches32, discard final16 outputs. Model has no batch normalization or cross-example operations.',
        'diagnostics':'FP32, UINT4/UINT8 per-layer-scale numerical controls, ideal8 programmed-mean network, and per-device-condition programmed-mean networks. Mean-weight network is not expectation of a noisy ReLU network. Oracle knows true parameters and is diagnostic only.',
        'decision':'All750 native full-test evaluations retained; no selection or success gate. Primary contrast cal4096 versus uncalibrated atT256; other T/P settings are predeclared secondary comparisons. Intervals across five trained-model means after averaging ten joint device/sampling realizations.',
        'test_scope':'These standard held-out-from-training test images were already used in previous experiments. Conditions fixed before nonideal scores; this is not a fresh test reservation, broader dataset confirmation or a training study.',
        'input_sha256':sha(DATA/'evaluation_data.npz'),'model_exports_sha256':{str(m):sha(DATA/f'seed{m}.npz') for m in range(5)},
        'source_sha256':{str(Path('research/thrml_spw')/n):sha(Path('research/thrml_spw')/n) for n in ['calibrated_mnist.py','controlled_positive_native.py','positive_inference.py','native.py','calibration_count_native.py','gain_calibration_math.py','calibration_control_math.py']},
        'scope':'Official THRML full learned positive-weight networks on all10000 MNIST test images. Static device errors and finite controls; external floating neural arithmetic, not physical hardware energy/latency or one total storage bit per weight.'}
    pp=OUT/'protocol.json'
    if pp.exists():assert json.loads(pp.read_text())==protocol
    else:pp.write_text(json.dumps(protocol,indent=2)+'\n')
    evaluation=np.load(DATA/'evaluation_data.npz');x=evaluation['test_inputs'];labels=evaluation['test_labels']
    assert x.shape==(10000,64) and x.dtype==np.float32
    padded=np.pad(x,((0,16),(0,0))).reshape(313,32,64);xb=jnp.asarray(padded)
    np.savez(OUT/'evaluation_identity.npz',example_indices=np.arange(10000),labels=labels)
    baseline_rows=[];mean_rows=[];rows=[];device_rows=[];controls={};started=time.monotonic()
    for model in range(5):
        export=np.load(DATA/f'seed{model}.npz');w=[export[f'weight_{i}'] for i in range(2)];bias=[export[f'bias_{i}'] for i in range(2)]
        reference=forward(x,w,bias);np.testing.assert_allclose(reference,export['reference_test_logits'],atol=1e-4)
        net=ControlledPositiveNetwork(w,bias);bank=net.calibration_bank;nf=len(bank.free_indices)
        base_codes=np.rint((np.clip(bank.biases.astype(float),-4,4)+4)*255/8).astype(np.uint8);b0=decode_command(base_codes)
        controls[str(model)]={'logical_weight_nodes':sum(a.size for a in w),'free_nodes':nf,'clamped_nodes':len(bank.clamped_indices),
            'free_by_layer':[len(l.bank.free_indices) for l in net.layers],'neural_biases':sum(a.size for a in bias),
            'nominal_range_clipped_fields':int(np.count_nonzero(np.abs(bank.biases)>4)),
            'one_method_command_bytes':nf,'endpoint_mode_bits_if_packed':2*sum(a.size for a in w),
            'actual_nominal_float32_field_bytes':bank.biases.nbytes,'affine_range_metadata_values':0}
        def mean_network(fields):
            qw=[];start=0
            for layer,weight in zip(net.layers,w):
                n=len(layer.bank.free_indices);a=weight.astype(float).copy().reshape(-1)
                a[layer.bank.free_indices]=np.exp(-np.logaddexp(0,-2*fields[start:start+n].astype(float)))
                qw.append(a.reshape(weight.shape));start+=n
            return forward(x,qw,bias)
        baselines={'FP32':reference,'programmed_mean_ideal8':mean_network(b0)}
        for bits in [4,8]:
            quant=[]
            for weight in w:
                scale=float(weight.max())/((1<<bits)-1);quant.append((np.rint(weight/scale).clip(0,(1<<bits)-1)*scale).astype(np.float32) if scale else weight.copy())
            baselines[f'UINT{bits}_per_layer_scale']=forward(x,quant,bias)
        np.savez_compressed(OUT/'baselines'/f'model{model}.npz',**baselines)
        for condition,logits in baselines.items():baseline_rows.append({'model':model,'condition':condition,**scores(logits,labels,reference)})
        funcs={t:net.dataset(t) for t in [64,256,1024]};counts={n:net.counts(n) for n in [512,2048]}
        for device in range(10):
            device_name=f'model{model}_device{device}';jp,npz=OUT/'devices'/f'{device_name}.json',OUT/'devices'/f'{device_name}.npz'
            noise=np.random.default_rng(2026110100+100*model+device).standard_normal((2,nf));offset=.2*noise[0];gain=np.exp(.2*noise[1]-.02)
            if jp.exists():
                devmeta=json.loads(jp.read_text());assert sha(npz)==devmeta['archive_sha256'] and devmeta['gain_sha256']==digest(gain) and devmeta['offset_sha256']==digest(offset)
                bundle=dict(np.load(npz))
            else:
                bundle={'gain':gain,'offset':offset,'nominal_codes':base_codes};calprops={}
                for p in [1024,4096]:
                    cc={}
                    for code in [96,159]:
                        probe=float(decode_command(np.array(code,np.uint8)));real=(gain*probe+offset).astype(np.float32)
                        key=jax.random.key(1080731)
                        for v in [model,device,p,code]:key=jax.random.fold_in(key,v)
                        cc[code]=np.asarray(counts[p//2](key,jnp.asarray(real)));bundle[f'count_P{p}_code{code}']=cc[code]
                    low,high=[float(decode_command(np.array(c,np.uint8))) for c in [96,159]]
                    ah,dh,raw=fit_two_probe(cc[96],cc[159],p//2,low,high);cmd,req=corrected_codes(b0,ah,dh)
                    bundle[f'codes_cal{p}']=cmd
                    calprops[str(p)]={'gain_rmse':float(np.sqrt(np.mean((ah-gain)**2))),'offset_rmse':float(np.sqrt(np.mean((dh-offset)**2))),
                        'gain_guard_low':int(np.count_nonzero(raw<.25)),'gain_guard_high':int(np.count_nonzero(raw>4)),
                        'bound_clipped_commands':int(np.count_nonzero(np.abs(req)>4)),'native_probe_runs':2,'free_evaluations':int(p*nf)}
                bundle['codes_oracle'],_=corrected_codes(b0,gain,offset)
                for condition in CONDITIONS:
                    codes=base_codes if condition in ['ideal8','uncalibrated'] else bundle[f'codes_{condition}']
                    bundle[f'fields_{condition}']=b0.copy() if condition=='ideal8' else (gain*decode_command(codes).astype(float)+offset).astype(np.float32)
                np.savez_compressed(npz,**bundle)
                devmeta={'model':model,'device':device,'gain_sha256':digest(gain),'offset_sha256':digest(offset),'archive_sha256':sha(npz),'calibration':calprops,
                    'field_hashes':{c:digest(bundle[f'fields_{c}']) for c in CONDITIONS},'native_probe_runs':4,'calibration_free_evaluations':int((1024+4096)*nf)}
                jp.write_text(json.dumps(devmeta,indent=2)+'\n')
            device_rows.append(devmeta)
            means={c:mean_network(bundle[f'fields_{c}']) for c in CONDITIONS};np.savez_compressed(OUT/'devices'/f'{device_name}_means.npz',**means)
            for condition,logits in means.items():mean_rows.append({'model':model,'device':device,'condition':condition,**scores(logits,labels,reference)})
            for t in [64,256,1024]:
                key=jax.random.key(1080730)
                for v in [model,device,t]:key=jax.random.fold_in(key,v)
                for condition in CONDITIONS:
                    name=f'{device_name}_{condition}_T{t}';rj,rn=OUT/'runs'/f'{name}.json',OUT/'runs'/f'{name}.npz'
                    fields=bundle[f'fields_{condition}']
                    if rj.exists():
                        row=json.loads(rj.read_text());assert row['field_sha256']==digest(fields) and sha(rn)==row['archive_sha256'];rows.append(row);continue
                    pre,logits=[np.asarray(a).reshape(-1,a.shape[-1])[:10000] for a in funcs[t](key,xb,jnp.asarray(fields))]
                    np.savez_compressed(rn,preactivation0=pre,logits=logits)
                    row={'model':model,'device':device,'condition':condition,'T':t,**scores(logits,labels,reference),
                        'accuracy_difference_vs_FP32':float(np.mean(logits.argmax(1)==labels)-np.mean(reference.argmax(1)==labels)),
                        'field_sha256':digest(fields),'device_archive_sha256':sha(npz),'archive_sha256':sha(rn),
                        'examples':10000,'batches':313,'free_node_evaluations':int(313*t*nf),'logical_weight_readouts':int(313*t*sum(a.size for a in w)),
                        'nominal_binary_products_excluding_padding':int(10000*t*sum(a.size for a in w)),
                        'nominal_binary_products_including_padding':int(10016*t*sum(a.size for a in w))}
                    rj.write_text(json.dumps(row,indent=2)+'\n');rows.append(row)
            print(f'Completed model{model} device{device}: '+', '.join(f'{c} T256={next(r["accuracy"] for r in rows if r["model"]==model and r["device"]==device and r["condition"]==c and r["T"]==256):.4f}' for c in CONDITIONS)+f'; elapsed={time.monotonic()-started:.1f}s',flush=True)
        jax.clear_caches()
    assert len(rows)==750 and len(device_rows)==50
    for file,rr in [('baseline_metrics.csv',baseline_rows),('mean_network_metrics.csv',mean_rows),('native_metrics.csv',rows)]:
        with (OUT/file).open('w',newline='') as f:
            wr=csv.DictWriter(f,fieldnames=list(rr[0]));wr.writeheader();wr.writerows(rr)
    result={'native_full_test_runs':750,'trained_models':5,'device_sampling_realizations_per_model':10,'native_calibration_probe_runs':200,
        'probability_mapping':'direct p=w, endpoints clamped, eight-bit finite fields, no affine ranges',
        'model_controls':controls,'executed_calibration_free_evaluations':int(sum(r['calibration_free_evaluations'] for r in device_rows)),
        'executed_inference_free_evaluations':int(sum(r['free_node_evaluations'] for r in rows)),
        'elapsed_seconds':time.monotonic()-started,'scope':protocol['scope']}
    (OUT/'REPORT.json').write_text(json.dumps(result,indent=2)+'\n');manifest(OUT/'manifest.json',protocol,result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
