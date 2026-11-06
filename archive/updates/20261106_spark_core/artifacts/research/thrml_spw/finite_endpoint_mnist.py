"""Fixed full-network study removing exact endpoint clamps at finite control range."""
import csv
import hashlib
import json
import time
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from .controlled_positive_native import ControlledPositiveNetwork
from .finite_endpoint_control import ENCODINGS,encoded_weights,quantize,decode
from .gain_calibration_math import fit_two_probe
from .calibrated_mnist import forward,scores
from .provenance import manifest
from .verify_transformer_control import sha

OUT=Path('results/thrml_spw/finite_endpoint_mnist');DATA=Path('data/thrml_spw/positive_mnist')
CONDITIONS=['ideal','uncalibrated','cal4096','oracle']


def digest(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def main():
    if jax.default_backend()!='gpu':raise RuntimeError('Pinned native study requires CUDA')
    OUT.mkdir(exist_ok=True)
    for folder in ['baselines','devices','runs']:(OUT/folder).mkdir(exist_ok=True)
    protocol={'schema':'spw-native-finite-endpoint-mnist-v1','encodings':ENCODINGS,'conditions':CONDITIONS,
        'model_seeds':list(range(5)),'device_sampling_seeds':list(range(10)),'samples_per_layer':[256,1024],
        'calibration_samples_per_free_device':4096,'test_examples':10000,'batch_size':32,
        'endpoint_rule':'Clamped control retains exact0/1 as before. Free variants clip target probabilities to sigmoid(-2*B),1-sigmoid(-2*B) in float64, so every weight has a free SpinNode. No affine range transform. Original learned FP32 network remains task reference.',
        'device_model':'Static positive a=exp(.2*z-.02), offset d~N(0,.2^2), realized field=float32(a*b_command+d); synthetic, not measured hardware. No drift.',
        'device_rng':'default_rng(2026110200+100*model_seed+device_seed).standard_normal((2,2368)); row0*.2 offset, exp(.2*row1-.02) gain. Slice original full coordinate vectors by each encoding free-index map, preserving device pairing across clamp/free geometry.',
        'calibration':'Nearest finite codes to command probes-1,+1.2048 official THRML samples per probe on every active free device; same smoothed-logit line fit and gain guard[.25,4]. Corrected inverse rounded/clipped to tested controller grid. No true parameters or task labels enter measured fit.',
        'keys':'Inference key1090730 fold model_seed,device_seed,T then batch; split layers. Same key seed across encodings/conditions, but clamp/free state dimensions differ, so no identical trajectory pairing across geometries is claimed. Probe key1090731 fold model_seed,device_seed,encoding_index,probe_index.',
        'execution':'313 batches32 with16 discarded padding rows. Actual THRML state output sums before first ReLU. Vmap independent batch chains is software parallelism. Joint calibration graph and layer graphs use the same device-coordinate parameters in separate modes.',
        'decision':'Execute all1600 full-test cases and400 probe windows; no selection, quality gate or significance-driven extension. Primary contrast free_B4_b8 versus clamped_B4_b8 atcal4096/T1024; wider-range/precision variants are predeclared secondary comparisons.',
        'cost_scope':'All variants have2368 weight coordinates, but all-free variants execute more active updates. Report those counts, actual uint8/uint16 command storage and logical8/10-bit payload separately. No physical control range/precision is assumed available.',
        'test_scope':'Existing standard held-out-from-training MNIST test images were used in earlier studies; this is not a fresh reservation. Five frozen positive-weight models, no retraining or test-driven tuning.',
        'input_sha256':sha(DATA/'evaluation_data.npz'),'model_exports_sha256':{str(m):sha(DATA/f'seed{m}.npz') for m in range(5)},
        'source_sha256':{str(Path('research/thrml_spw')/n):sha(Path('research/thrml_spw')/n) for n in ['finite_endpoint_mnist.py','finite_endpoint_control.py','controlled_positive_native.py','positive_inference.py','native.py','calibration_count_native.py','gain_calibration_math.py','calibrated_mnist.py']},
        'scope':'Complete learned direct-probability networks, finite bias endpoints with/without exact modes, synthetic static gain/offset errors, actual official THRML draws and external floating arithmetic. No physical, energy/latency or universal precision claim.'}
    pp=OUT/'protocol.json'
    if pp.exists():assert json.loads(pp.read_text())==protocol
    else:pp.write_text(json.dumps(protocol,indent=2)+'\n')
    d=np.load(DATA/'evaluation_data.npz');x,y=d['test_inputs'],d['test_labels'];xb=jnp.asarray(np.pad(x,((0,16),(0,0))).reshape(313,32,64))
    np.savez(OUT/'evaluation_identity.npz',labels=y,example_indices=np.arange(10000))
    rows=[];base_rows=[];mean_rows=[];device_rows=[];controls={};started=time.monotonic()
    for model in range(5):
        export=np.load(DATA/f'seed{model}.npz');original=[export[f'weight_{i}'] for i in range(2)];bias=[export[f'bias_{i}'] for i in range(2)]
        reference=forward(x,original,bias);np.testing.assert_allclose(reference,export['reference_test_logits'],atol=1e-4)
        classical={'FP32':reference}
        for bits in [4,8]:
            qw=[]
            for w in original:
                scale=float(w.max())/((1<<bits)-1);qw.append((np.rint(w/scale).clip(0,(1<<bits)-1)*scale).astype(np.float32) if scale else w.copy())
            classical[f'UINT{bits}']=forward(x,qw,bias)
        np.savez_compressed(OUT/'baselines'/f'model{model}.npz',**classical)
        for condition,value in classical.items():base_rows.append({'model':model,'condition':condition,**scores(value,y,reference)})
        controls[str(model)]={}
        for ei,encoding in enumerate(ENCODINGS):
            en=encoding['name'];bound,bits=encoding['bound'],encoding['bits'];weights=encoded_weights(original,encoding)
            net=ControlledPositiveNetwork(weights,bias);bank=net.calibration_bank;active=bank.free_indices;nf=len(active)
            if not encoding['clamp_endpoints']:assert nf==2368 and len(bank.clamped_indices)==0
            code0=quantize(bank.biases,bits,bound);b0=decode(code0,bits,bound)
            probes=[quantize(np.array(v),bits,bound) for v in [-1.,1.]];pf=[float(decode(c,bits,bound)) for c in probes]
            controls[str(model)][en]={'logical_nodes':2368,'free_nodes':nf,'clamped_nodes':len(bank.clamped_indices),
                'logical_command_payload_bits':nf*bits,'actual_command_array_bytes':code0.nbytes,
                'endpoint_mode_bits_if_packed':4736 if encoding['clamp_endpoints'] else 0,
                'neural_bias_values':42,'actual_base_field_bytes':bank.biases.nbytes,'free_by_layer':[len(l.bank.free_indices) for l in net.layers],
                'ideal_endpoint_probability_floor':float(1/(1+np.exp(2*bound))),'probe_codes':[int(c) for c in probes],'probe_fields':pf}
            def mean_network(fields):
                qw=[];offset=0
                for w,layer in zip(weights,net.layers):
                    n=len(layer.bank.free_indices);a=w.reshape(-1).copy()
                    a[layer.bank.free_indices]=np.exp(-np.logaddexp(0,-2*fields[offset:offset+n].astype(float)))
                    qw.append(a.reshape(w.shape));offset+=n
                return forward(x,qw,bias)
            fns={t:net.dataset(t) for t in [256,1024]};counter=net.counts(2048)
            for device in range(10):
                name=f'model{model}_device{device}_{en}';jp,npz=OUT/'devices'/f'{name}.json',OUT/'devices'/f'{name}.npz'
                noise=np.random.default_rng(2026110200+100*model+device).standard_normal((2,2368));gain_full=np.exp(.2*noise[1]-.02);delta_full=.2*noise[0]
                gain,delta=gain_full[active],delta_full[active]
                if jp.exists():
                    meta=json.loads(jp.read_text());assert sha(npz)==meta['archive_sha256'] and meta['gain_sha256']==digest(gain) and meta['offset_sha256']==digest(delta)
                    bundle=dict(np.load(npz))
                else:
                    bundle={'nominal_codes':code0,'active_indices':active,'gain':gain,'offset':delta};cc=[]
                    for pi,field in enumerate(pf):
                        real=(gain*field+delta).astype(np.float32);key=jax.random.key(1090731)
                        for v in [model,device,ei,pi]:key=jax.random.fold_in(key,v)
                        c=np.asarray(counter(key,jnp.asarray(real)));bundle[f'count_{pi}']=c;cc.append(c)
                    ah,dh,raw=fit_two_probe(cc[0],cc[1],2048,*pf)
                    requested=(b0.astype(float)-dh)/ah;bundle['codes_cal4096']=quantize(requested,bits,bound)
                    bundle['codes_oracle']=quantize((b0.astype(float)-delta)/gain,bits,bound)
                    for condition in CONDITIONS:
                        codes=code0 if condition in ['ideal','uncalibrated'] else bundle[f'codes_{condition}']
                        bundle[f'fields_{condition}']=b0.copy() if condition=='ideal' else (gain*decode(codes,bits,bound).astype(float)+delta).astype(np.float32)
                    np.savez_compressed(npz,**bundle)
                    meta={'model':model,'device':device,'encoding':en,'archive_sha256':sha(npz),'gain_sha256':digest(gain),'offset_sha256':digest(delta),
                        'field_hashes':{c:digest(bundle[f'fields_{c}']) for c in CONDITIONS},
                        'gain_rmse':float(np.sqrt(np.mean((ah-gain)**2))),'offset_rmse':float(np.sqrt(np.mean((dh-delta)**2))),
                        'gain_guard_low':int(np.count_nonzero(raw<.25)),'gain_guard_high':int(np.count_nonzero(raw>4)),
                        'bound_clipped_corrected_commands':int(np.count_nonzero(np.abs(requested)>bound)),
                        'native_probe_runs':2,'calibration_free_evaluations':int(4096*nf)}
                    jp.write_text(json.dumps(meta,indent=2)+'\n')
                device_rows.append(meta)
                means={c:mean_network(bundle[f'fields_{c}']) for c in CONDITIONS};np.savez_compressed(OUT/'devices'/f'{name}_means.npz',**means)
                for c,logits in means.items():mean_rows.append({'model':model,'device':device,'encoding':en,'condition':c,**scores(logits,y,reference)})
                for t in [256,1024]:
                    key=jax.random.key(1090730)
                    for v in [model,device,t]:key=jax.random.fold_in(key,v)
                    for condition in CONDITIONS:
                        stem=f'{name}_{condition}_T{t}';rj,rn=OUT/'runs'/f'{stem}.json',OUT/'runs'/f'{stem}.npz';fields=bundle[f'fields_{condition}']
                        if rj.exists():
                            row=json.loads(rj.read_text());assert sha(rn)==row['archive_sha256'] and row['field_sha256']==digest(fields);rows.append(row);continue
                        pre,logits=[np.asarray(a).reshape(-1,a.shape[-1])[:10000] for a in fns[t](key,xb,jnp.asarray(fields))]
                        np.savez_compressed(rn,preactivation0=pre,logits=logits)
                        row={'model':model,'device':device,'encoding':en,'condition':condition,'T':t,**scores(logits,y,reference),
                            'accuracy_difference_vs_FP32':float(np.mean(logits.argmax(1)==y)-np.mean(reference.argmax(1)==y)),
                            'archive_sha256':sha(rn),'field_sha256':digest(fields),'device_archive_sha256':sha(npz),
                            'free_node_evaluations':int(313*t*nf),'logical_weight_readouts':313*t*2368,
                            'binary_products_with_padding':10016*t*2368,'binary_products_excluding_padding':10000*t*2368}
                        rj.write_text(json.dumps(row,indent=2)+'\n');rows.append(row)
                print(f'Completed model{model} {en} device{device}: cal4096 T1024={next(r["accuracy"] for r in rows if r["model"]==model and r["device"]==device and r["encoding"]==en and r["condition"]=="cal4096" and r["T"]==1024):.4f}; elapsed={time.monotonic()-started:.1f}s',flush=True)
            jax.clear_caches()
    assert len(rows)==1600 and len(device_rows)==200
    for filename,rr in [('native_metrics.csv',rows),('baseline_metrics.csv',base_rows),('mean_network_metrics.csv',mean_rows)]:
        with (OUT/filename).open('w',newline='') as f:
            wr=csv.DictWriter(f,fieldnames=list(rr[0]));wr.writeheader();wr.writerows(rr)
    result={'native_full_test_runs':1600,'native_probe_runs':400,'model_controls':controls,
        'executed_calibration_free_evaluations':int(sum(r['calibration_free_evaluations'] for r in device_rows)),
        'executed_inference_free_evaluations':int(sum(r['free_node_evaluations'] for r in rows)),
        'elapsed_seconds':time.monotonic()-started,'scope':protocol['scope']}
    (OUT/'REPORT.json').write_text(json.dumps(result,indent=2)+'\n');manifest(OUT/'manifest.json',protocol,result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
