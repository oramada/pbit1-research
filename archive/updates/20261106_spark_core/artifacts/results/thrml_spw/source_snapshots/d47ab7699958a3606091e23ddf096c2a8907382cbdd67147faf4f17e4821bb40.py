"""Independent format rounding, functional graph and paired statistics audit."""
import argparse,json,datetime
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from results.thrml_spw.cifar_cnn_native.precision_baselines.common import ROOT,OUT,TRAIN,PARENT,NAMES,sha,write,guard,parent_closed
from results.thrml_spw.fashion_cnn_native.cohort.common import numpy_control
def quantize(a,name):
    a=np.asarray(a,np.float32)
    if name.startswith('FP16'):return a.astype(np.float16).astype(np.float64)
    bits=a.view(np.uint32);rounded=(bits+np.uint32(0x7fff)+((bits>>np.uint32(16))&np.uint32(1)))&np.uint32(0xffff0000)
    return rounded.view(np.float32).astype(np.float64)
def independent(x,model,name):
    weights=[quantize(model[p+'_weight'],name) for p in ['conv1','conv2','head']]
    if name.endswith('weights_F64'):return numpy_control(x,model,weights,False)
    dtype=torch.float16 if name.startswith('FP16') else torch.bfloat16;h=torch.as_tensor(quantize(x,name),dtype=dtype)
    with torch.no_grad():
        for li,(prefix,w) in enumerate(zip(['conv1','conv2','head'],weights)):
            ww=torch.as_tensor(w,dtype=dtype);bb=torch.as_tensor(quantize(model[prefix+'_bias'],name),dtype=dtype)
            if li<2:h=F.max_pool2d(F.relu(F.conv2d(h,ww,bb,padding=1)),2)
            else:h=F.linear(h.flatten(1),ww,bb)
    assert h.dtype==dtype;return h.float().numpy().astype(float)
def metric(z,y):
    shifted=z-z.max(1)[:,None];loss=np.log(np.exp(shifted).sum(1))-shifted[np.arange(len(y)),y];return np.array([np.count_nonzero(z.argmax(1)==y)/len(y),loss.mean()])
def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['selfcheck','run']);args=parser.parse_args();torch.set_num_threads(4);torch.use_deterministic_algorithms(True);train=json.loads((TRAIN/'manifest.json').read_text())
    if args.action=='selfcheck':
        from results.thrml_spw.cifar_cnn_native.precision_baselines.common import output
        path=ROOT/'precision_selfcheck.json';assert not path.exists();x=np.load(TRAIN/'validation.npz')['inputs'][:64];rows=[]
        for r in train['rows']:
            model=np.load(r['path'])
            for name in NAMES:
                # Independently rounded bits must match the actual Torch conversion.
                dtype=torch.float16 if name.startswith('FP16') else torch.bfloat16
                for prefix in ['conv1','conv2','head']:
                    for suffix in ['weight','bias']:
                        raw=model[prefix+'_'+suffix];np.testing.assert_array_equal(quantize(raw,name),torch.from_numpy(raw).to(dtype).float().numpy().astype(float))
                z=output(x,r,name);ref=independent(x,model,name)
                if name.endswith('Torch_CPU'):np.testing.assert_array_equal(z,ref)
                else:np.testing.assert_allclose(z,ref,rtol=2e-12,atol=2e-11)
                np.testing.assert_array_equal(z.argmax(1),ref.argmax(1));rows.append({'model':r['seed'],'condition':name,'max_error':float(np.max(abs(z-ref)))})
        write(path,{'passed':True,'rows':rows,'source_sha256':{str(f):sha(f) for f in sorted((ROOT/'precision_baselines').glob('*.py'))},'scope':'20 validation-only paths64inputs each; no labels/scoring. FP16 NumPy conversion/BF16 IEEE RNE bit rounding check weight/bias conversions; functional low-precision Torch graph versus exported module graph. Storage-only F64 uses independent NumPy spatial contractions. No claimed proof of internal accumulator precision.'});print('All20 floating-format validation paths pass',flush=True);return
    p=json.loads((OUT/'protocol.json').read_text());guard(p);parent_closed(p);e=json.loads((OUT/'evaluation.json').read_text());s=json.loads((OUT/'statistics.json').read_text());assert e['logits_sha256']==sha(OUT/'logits.npz') and e['test_data_sha256']==sha(PARENT/'test_data.npz') and e['protocol_sha256']==sha(OUT/'protocol.json');assert s['parent_statistics_sha256']==sha(PARENT/'statistics.json') and s['parent_statistical_arrays_sha256']==sha(PARENT/'statistical_arrays.npz') and s['evaluation_sha256']==sha(OUT/'evaluation.json') and s['logits_sha256']==sha(OUT/'logits.npz') and s['statistical_arrays_sha256']==sha(OUT/'statistical_arrays.npz');d=np.load(PARENT/'test_data.npz');x,y=d['inputs'],d['labels'];saved=np.load(OUT/'logits.npz');arrays=np.load(OUT/'statistical_arrays.npz');parent=np.load(PARENT/'statistical_arrays.npz');base=np.empty((4,5,2));maxerr={}
    for mi,r in enumerate(train['rows']):
        model=np.load(r['path'])
        for ci,name in enumerate(NAMES):
            ref=np.concatenate([independent(x[first:first+128],model,name) for first in range(0,10000,128)]);z=saved[f'c{ci}_model{mi}']
            if name.endswith('Torch_CPU'):np.testing.assert_array_equal(z,ref)
            else:np.testing.assert_allclose(z,ref,rtol=2e-12,atol=2e-11)
            np.testing.assert_array_equal(z.argmax(1),ref.argmax(1));base[ci,mi]=metric(z,y);maxerr[f'model{mi}_{name}']=float(np.max(abs(z-ref)));print('Verified',mi,name,flush=True)
    np.testing.assert_allclose(base,arrays['control_metrics'],rtol=2e-12,atol=2e-13);bm,bs=parent['bootstrap_models'],parent['bootstrap_seeds']
    assert len(s['contrasts'])==12 and {(c['budget_index'],c['control_index']) for c in s['contrasts']}=={(bi,ci) for bi in range(3) for ci in range(4)}
    for c in s['contrasts']:
        bi,ci=c['budget_index'],c['control_index'];assert c['control']==NAMES[ci] and c['T']==p['samples'][bi];delta=parent['values'][bi]-base[ci,:,None,:];z=sum(delta[bm[:,i,None],bs[:,i,:]].sum(1) for i in range(5))/50;np.testing.assert_allclose(z,arrays[f'budget{bi}_control{ci}'],rtol=2e-10,atol=2e-12);np.testing.assert_allclose(delta.mean((0,1)),c['difference'],rtol=2e-10,atol=2e-12);np.testing.assert_allclose(np.quantile(z,[.025,.975],axis=0).T,c['descriptive_95_interval'],rtol=2e-10,atol=2e-12)
    for ci,name in enumerate(NAMES):np.testing.assert_allclose(base[ci].mean(0),s['summary'][name],rtol=2e-12,atol=2e-13)
    guard(p);write(OUT/'evidence_audit.json',{'passed':True,'model_controls_checked':20,'full_logits_checked':2000000,'bootstrap_values_checked':120000,'control_max_errors':maxerr,'new_native_updates':0,'statistics_sha256':sha(OUT/'statistics.json')})
    lines=['# Conventional FP16/BF16 controls for the native CNN cohort','','Four prespecified numerical controls on all five frozen CNNs/full10000historically consumed images. No new p-bit draws. Weight-storage controls round only kernels/head weights toFP16/BF16 and execute F64 arithmetic with originalFP32biases. CPU execution controls convert inputs/weights/biases to the selected dtype and use the pinned TorchCPUbackend; observed output dtype is checked. This does not prove internal accumulator width or physical hardware arithmetic.','','| Control | Mean accuracy (%) | Mean CE |','|---|---:|---:|']
    for name,v in s['summary'].items():lines.append(f'|{name}|{100*v[0]:.4f}|{v[1]:.8f}|')
    lines+=['','Independent NumPy FP16 conversion/BF16 RNE bit rounding checks the format mapping. Functional Torch graphs replay the full low-precision module graph exactly; storage-only controls use independent NumPyF64spatial contractions within2e-11absolute/2e-12relative tolerance. All12native-minus-format contrasts reuse the parent five-model/ten-seed5000bootstrapdraws on fixed consumed images. These are descriptive intervals, not independent MC replications or fresh generalization confirmation.','','Nominal weight storage is7856bytes/model for either16-bitformat; weight-only controls retain136biasbytes, whole-low-precision controls68biasbytes. Activations/control/readout/framework buffers and hardware costs remain separate. No measured energy/latency/area win or full-roadmap completion follows.','','[Protocol](cifar_cnn_native/precision_evidence/protocol.json) · [All contrasts](cifar_cnn_native/precision_evidence/statistics.json) · [Audit](cifar_cnn_native/precision_evidence/evidence_audit.json)']
    report=Path('results/thrml_spw/CIFAR_CNN_PRECISION_REPORT.md');assert not report.exists();report.write_text('\n'.join(lines)+'\n');paths=[OUT/n for n in ['protocol.json','evaluation.json','logits.npz','statistics.json','statistical_arrays.npz','evidence_audit.json']]+[report];write(OUT/'completion.json',{'status':'FP16_BF16_RGB_controls_audited_goal_active','recorded_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'new_native_updates':0,'artifact_sha256':{str(f):sha(f) for f in paths}})
if __name__=='__main__':main()
