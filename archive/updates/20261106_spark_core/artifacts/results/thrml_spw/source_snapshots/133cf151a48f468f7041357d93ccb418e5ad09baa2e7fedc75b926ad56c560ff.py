"""Fixed all-free full-test drift and scheduled-refresh native THRML study."""
import csv,hashlib,json,time
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from .clocked_network_native import ClockedPositiveNetwork
from . import network_drift_clock as clock
from .finite_endpoint_control import encoded_weights,quantize,decode
from .gain_calibration_math import fit_two_probe
from .calibrated_mnist import forward,scores
from .provenance import manifest
from .verify_transformer_control import sha
OUT=Path('results/thrml_spw/clocked_mnist');DATA=Path('data/thrml_spw/positive_mnist')
POLICIES=('retain_initial','refresh_every32_batches')
ENC={'name':'free_B4_b8','bound':4.,'bits':8,'clamp_endpoints':False}


def digest(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def frame_hashes(a):return [digest(f) for f in a.reshape(-1,a.shape[-1])]


def main():
    if jax.default_backend()!='gpu':raise RuntimeError('CUDA native study required')
    OUT.mkdir(exist_ok=True)
    for folder in ['calibration','runs','baselines']:(OUT/folder).mkdir(exist_ok=True)
    files=['clocked_mnist.py','clocked_network_native.py','network_drift_clock.py','controlled_positive_native.py','finite_endpoint_control.py','gain_calibration_math.py','calibrated_mnist.py','positive_inference.py','native.py','calibration_count_native.py']
    protocol={'schema':'spw-clocked-all-free-full-mnist-v1','model_seeds':list(range(5)),'device_path_sampling_seeds':list(range(5)),
        'encoding':ENC,'samples_per_layer':1024,'segment_sweeps':64,'drift_SD_per_1024_sweeps':list(clock.RATES),
        'policies':list(POLICIES),'test_examples':10000,'batch_size':32,'batches':313,'refresh_batch_interval':32,'calibration_windows':10,
        'calibration_samples_per_free_node_per_window':4096,'total_serial_sweeps':681984,
        'device_model':'a=exp(.2*z-.02), initial d=.2*z on2368 full coordinates fromdefault_rng(2026110400+100*model+device), row0 offset_noise androw1 gain_noise. Drift is independent piecewise Brownian offset increments SD=rate/sqrt(16) per64-sweep segment, shared path across policies/rates. F64 path arithmetic emittedasF32 fields; no gain drift, temporal/cross-node correlation or readout noise.',
        'path_rng':'default_rng(2026110500+100*model+device), total_segments x2368 standard normals, cumulative from zero dividedby4. Value at left segment boundary used for64sweeps; drift advances for all nodes during both layers, probes and retained-policy idle.',
        'clock':'Ten4096-sweep windows before batches0,32,...288. Each batch executes layer0for1024sweeps thenlayer1for1024. Retained policy idles at nine later calibration windows, while refresh samples two probes; absolute querytimes/path therefore match. Layer-inactive nodes drift but no unobserved hardware-switchingenergy is modeled. Calibration computation/programmingtime excluded; no hardwareseconds claim.',
        'calibration':'Joint2368-node calibration graph; two32-segment probes at nearest8bit[-4,4] codes to-1,+1,2048draws each; drift changes every64sweeps inside each probe. Fit smoothed-logit line, gain guard[.25,4], inversegridclip. Retain uses firstwindowcodes; refreshusescurrentwindowcodes. Fit never receives true device parameters or labels.',
        'keys':'Countkey1100731foldmodel,device,window,thenabsolute-local-segment0..63. Inferkey1100730foldmodel,device,thenbatch,layer,segment0..15. Keys shared across rates/policies; free geometry fixed. Each unary segment initializedfalse, firstsweepalready samples independent marginal; actual64sweeps percall, no extra burn-in.',
        'inputs':'All10000 existing test examples in fixeddefault_rng(2026110301).permutation(10000) order;16zeropadded rows discarded. Shared order acrossconditions. Not freshdata. No training or score-dependent selection.',
        'execution':'One free native THRML SpinNode perweight, no clamp blocks. Segment native linear-output means averaged before first ReLU. Dataset/segment vmap softwareparallelism evaluates predeclared serial-clock paths, not physicalparallelcapacity. Resetting segment state models independent unary pbits only.',
        'decision':'Execute all150 complete test cases and750 calibration events (1500probe windows). No qualitygate, tuning or significance-driven extension. Primary contrast refreshminusretain at rate.04; zero-drift andrate.01 controls retained. Full-test and each chronological window results reported.',
        'cost':'Calibrationjoint2368 nodes, inference layer0=2048/layer1=320. Bothpolicies681984elapsednative-sweep clock. Retain4096activecalibration+36864idle; refresh40960activecalibration+0idle. Same641024layer-sampling sweeps, unequal active probe counts. No equality of physicalenergy/latency assumed.',
        'input_sha256':sha(DATA/'evaluation_data.npz'),'model_sha256':{str(m):sha(DATA/f'seed{m}.npz') for m in range(5)},
        'source_sha256':{str(Path('research/thrml_spw')/f):sha(Path('research/thrml_spw')/f) for f in files}}
    pp=OUT/'protocol.json'
    if pp.exists():assert json.loads(pp.read_text())==protocol
    else:pp.write_text(json.dumps(protocol,indent=2)+'\n')
    archive=Path('results/thrml_spw/source_snapshots');archive.mkdir(exist_ok=True)
    for name,h in protocol['source_sha256'].items():
        p=archive/(h+'.py');data=Path(name).read_bytes()
        if p.exists():assert p.read_bytes()==data
        else:p.write_bytes(data)
    d=np.load(DATA/'evaluation_data.npz');ids=np.random.default_rng(2026110301).permutation(10000);x,y=d['test_inputs'][ids],d['test_labels'][ids]
    np.savez(OUT/'evaluation_identity.npz',example_indices=ids,labels=y)
    xb=jnp.asarray(np.pad(x,((0,16),(0,0))).reshape(313,32,64))
    i0=np.array([clock.batch_start(b)+np.arange(clock.LAYER_SEGMENTS) for b in range(313)]);i1=i0+clock.LAYER_SEGMENTS
    rows=[];window_rows=[];baselines=[];cal_rows=[];started=time.monotonic()
    for model in range(5):
        e=np.load(DATA/f'seed{model}.npz');original=[e[f'weight_{i}'] for i in range(2)];biases=[e[f'bias_{i}'] for i in range(2)]
        weights=encoded_weights(original,ENC);net=ClockedPositiveNetwork(weights,biases);bank=net.calibration_bank
        assert len(bank.free_indices)==2368 and not len(bank.clamped_indices)
        split=net.free_split;assert split==2048
        nominal=decode(quantize(bank.biases,8,4),8,4);probe=[float(decode(quantize(np.array(v),8,4),8,4)) for v in [-1.,1.]]
        reference=forward(x,original,biases);np.testing.assert_allclose(reference,e['reference_test_logits'][ids],atol=1e-4)
        np.savez_compressed(OUT/'baselines'/f'model{model}.npz',FP32=reference)
        baselines.append({'model':model,'condition':'FP32',**scores(reference,y,reference)})
        infer=net.dataset_segments(64);count=net.segmented_counts(64)
        for device in range(5):
            path=clock.standard_path(model,device,2368);gain,initial=clock.static_device(model,device,2368)
            path_hash=digest(path)
            for rate in clock.RATES:
                tag=f'model{model}_device{device}_rate{rate:g}';cp=OUT/'calibration'/f'{tag}.npz';cj=cp.with_suffix('.json')
                if cj.exists():
                    cm=json.loads(cj.read_text());assert sha(cp)==cm['archive_sha256'] and path_hash==cm['path_sha256'];bundle=dict(np.load(cp))
                else:
                    counts=[];codes=[];hashes=[];gains=[];offsets=[];guards=[]
                    for window in range(clock.WINDOWS):
                        idx=clock.calibration_start(window)+np.arange(64)
                        commands=np.repeat(probe,32)[:,None]
                        fields=clock.field_schedule(gain,initial,path,idx,rate,commands)
                        key=jax.random.key(1100731)
                        for v in [model,device,window]:key=jax.random.fold_in(key,v)
                        cc=np.asarray(count(key,jnp.asarray(fields)));ah,dh,raw=fit_two_probe(cc[:32].sum(0),cc[32:].sum(0),2048,*probe)
                        code=quantize((nominal.astype(float)-dh)/ah,8,4)
                        counts.append(cc.astype(np.uint8));codes.append(code);hashes.append(frame_hashes(fields));gains.append(ah);offsets.append(dh)
                        guards.append({'low':int(np.count_nonzero(raw<.25)),'high':int(np.count_nonzero(raw>4))})
                    bundle={'segment_counts':np.stack(counts),'corrected_codes':np.stack(codes),'gain_estimates':np.stack(gains),'offset_estimates':np.stack(offsets),'gain':gain,'initial_offset':initial,'nominal_codes':quantize(bank.biases,8,4)}
                    np.savez_compressed(cp,**bundle);cm={'model':model,'device':device,'rate':rate,'archive_sha256':sha(cp),'path_sha256':path_hash,'calibration_field_hashes':hashes,'gain_guards':guards,'calibration_events':10,'probe_windows':20,'native_segment_calls':640,'free_node_evaluations':10*4096*2368}
                    cj.write_text(json.dumps(cm,indent=2)+'\n')
                cal_rows.append(cm)
                for policy in POLICIES:
                    name=f'{tag}_{policy}';rn=OUT/'runs'/f'{name}.npz';rj=rn.with_suffix('.json')
                    if rj.exists():
                        row=json.loads(rj.read_text());assert sha(rn)==row['archive_sha256'];rows.append(row);window_rows.extend(row['windows']);continue
                    indices=np.zeros(313,dtype=int) if policy=='retain_initial' else np.arange(313)//32
                    commands=decode(bundle['corrected_codes'][indices],8,4)
                    f0=clock.field_schedule(gain[:split],initial[:split],path[:,:split],i0,rate,commands[:,None,:split])
                    f1=clock.field_schedule(gain[split:],initial[split:],path[:,split:],i1,rate,commands[:,None,split:])
                    key=jax.random.key(1100730)
                    for v in [model,device]:key=jax.random.fold_in(key,v)
                    pre,logits=[np.asarray(v).reshape(-1,v.shape[-1])[:10000] for v in infer(key,xb,jnp.asarray(f0),jnp.asarray(f1))]
                    np.savez_compressed(rn,preactivation0=pre,logits=logits)
                    wr=[]
                    for window in range(10):
                        low,high=window*1024,min((window+1)*1024,10000)
                        wr.append({'model':model,'device':device,'rate':rate,'policy':policy,'window':window,'examples':high-low,**scores(logits[low:high],y[low:high],reference[low:high])})
                    row={'model':model,'device':device,'rate':rate,'policy':policy,**scores(logits,y,reference),'archive_sha256':sha(rn),'calibration_archive_sha256':sha(cp),'path_sha256':path_hash,
                        'layer_field_hashes':[frame_hashes(f0),frame_hashes(f1)],'inference_free_evaluations':313*1024*2368,
                        'candidate_calibration_free_evaluations':4096*2368*(1 if policy=='retain_initial' else 10),
                        'inference_serial_sweeps':313*2048,'active_calibration_sweeps':4096*(1 if policy=='retain_initial' else 10),
                        'idle_sweeps':4096*(9 if policy=='retain_initial' else 0),'elapsed_sweep_clock':681984,'logical_weight_readouts':313*1024*2368,
                        'binary_products_with_padding':10016*1024*2368,'binary_products_without_padding':10000*1024*2368,'windows':wr}
                    rj.write_text(json.dumps(row,indent=2)+'\n');rows.append(row);window_rows.extend(wr)
                    print(f'{name}: accuracy={row["accuracy"]:.5f}, elapsed={time.monotonic()-started:.1f}s',flush=True)
                del f0,f1
            del path
        jax.clear_caches()
    assert len(rows)==150 and len(cal_rows)==75 and len(window_rows)==1500
    for filename,data in [('native_metrics.csv',[{k:v for k,v in r.items() if k not in ['layer_field_hashes','windows']} for r in rows]),('window_metrics.csv',window_rows),('baseline_metrics.csv',baselines)]:
        with (OUT/filename).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    result={'native_full_test_runs':len(rows),'native_calibration_events':750,'native_probe_windows':1500,'calibration_segment_calls':48000,
        'inference_layer_segment_calls':150*313*2*16,'production_inference_free_evaluations':sum(r['inference_free_evaluations'] for r in rows),
        'production_calibration_free_evaluations':sum(r['free_node_evaluations'] for r in cal_rows),'elapsed_seconds':time.monotonic()-started,
        'scope':'All-free finite endpoint full networks with explicit synthetic drift during probes/inference/idle. Counterfactual matched clocks; counts are observed native updates, not physical switching energy. Static gains, independent unary transitions, old test set.'}
    (OUT/'REPORT.json').write_text(json.dumps(result,indent=2)+'\n');manifest(OUT/'manifest.json',protocol,result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
