"""Audit full unique parameter coverage, saved losses and node-work receipts."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    root=Path('results/thrml_spw/lfm3_1p7b');out=root/'group128_T256'
    protocol=json.loads((out/'protocol.json').read_text());r=json.loads((out/'results.json').read_text())
    inventory=json.loads((out/'conversion_inventory.json').read_text())
    meta=json.loads((root/'meta_inventory.json').read_text())
    for source,key in [('group128_T256.py','source_sha256'),('group128_worker.py','worker_sha256')]:
        assert sha(Path(__file__).with_name(source))==protocol[key]
    assert sha(out/'outputs.npz')==r['outputs_sha256']
    assert sha(root/'baseline_smoke.npz')==protocol['input_sha256']
    expected_parameters={p['name']:p for p in meta['parameters']}
    canonical={}
    modules={m['name']:m for m in inventory['modules']}
    for m in modules.values():
        name='model.embed_tokens' if m['shared_coordinate_group']=='token_embedding_head' else m['name']
        key=name+'.weight'
        assert key in expected_parameters
        assert m['shape']==expected_parameters[key]['shape']
        canonical[key]=expected_parameters[key]
    assert set(canonical)==set(expected_parameters)
    total=sum(p['numel'] for p in canonical.values())
    assert total==r['unique_parameters_converted']==r['total_parameters']==1728801536
    tied=[m['name'] for m in modules.values() if m['shared_coordinate_group']=='token_embedding_head']
    assert sorted(tied)==['lm_head','model.embed_tokens']
    a=np.load(out/'outputs.npz');ids=a['input_ids'];baseline=a['BF16']
    np.testing.assert_array_equal(baseline,np.load(root/'baseline_smoke.npz')['logits'])
    np.testing.assert_array_equal(baseline,a['identity_bridge'])
    losses=[]
    for metric in r['metrics']:
        y=a[metric['stage']];assert np.isfinite(y).all() and y.shape==baseline.shape
        # Independent Torch double log-softmax, versus producer NumPy logsumexp.
        ce=F.cross_entropy(torch.from_numpy(y[:,:-1].astype(np.float64).reshape(-1,y.shape[-1])),torch.from_numpy(ids[:,1:].reshape(-1))).item()
        np.testing.assert_allclose(metric['diagnostic_CE_FP64'],ce,rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(metric['logit_MSE_vs_BF16'],np.mean((y.astype(np.float64)-baseline)**2),rtol=1e-12,atol=1e-12)
        assert metric['argmax_agreement']==float(np.mean(y.argmax(-1)==baseline.argmax(-1)))
        losses.append({'stage':metric['stage'],'diagnostic_CE_FP64':ce})
    retained=0;warmup=0;drawn=0
    for stage in ['nominal_controls','native']:
        calls=[c for c in r['operator_calls'] if c['stage']==stage]
        assert len(calls)==len(modules) and {c['module'] for c in calls}==set(modules)
        for c in calls:
            m=modules[c['module']];kind=m['kind'];shape=m['shape']
            assert c['kind']==kind and c['finite']
            if kind=='embedding':
                assert c['embedding_rows']==np.unique(ids).tolist()
                fanin=1;channels=len(c['embedding_rows'])*shape[1]
            elif kind=='elementwise':fanin=1;channels=shape[0]
            elif kind=='conv':fanin=shape[2];channels=shape[0]
            else:fanin=shape[1];channels=shape[0]
            n=fanin*channels;pad=fanin*((channels+63)//64*64-channels)
            assert c['weight_coordinates']==n and c['padding_nodes']==pad
            if stage=='native':
                assert c['retained_evaluations']==protocol['samples']*(n+pad)
                assert c['warmup_evaluations']==n+pad
                retained+=c['retained_evaluations'];warmup+=c['warmup_evaluations'];drawn+=n
            else:assert c['retained_evaluations']==0
    prior=np.load(root/'all_weight_T256/outputs.npz')
    for name in ['BF16','identity_bridge']:np.testing.assert_array_equal(a[name],prior[name])
    assert protocol['samples']==256
    previous=json.loads((root/'all_weight_T256/independent_audit.json').read_text())
    assert retained==previous['retained_evaluations_including_padding'] and warmup==previous['warmup_evaluations_including_padding']
    audit={'matched_row_T256_node_work_exact':True,'matched_BF16_controls_exact':True,'samples':protocol['samples'],'status':'all_unique_parameters_and_saved_metrics_verified','source_sha256':sha(Path(__file__)),
        'results_sha256':sha(out/'results.json'),'unique_parameters_converted':total,'coverage_fraction':1.0,
        'operator_modules_per_forward':len(modules),'sampled_coordinates_across_uses':drawn,
        'retained_evaluations_including_padding':retained,'warmup_evaluations_including_padding':warmup,
        'losses':losses,'scope':'Coverage/metric audit of one full-parameter native forward and controls. Tied embeddings count once, sampled uses count separately. Exact operator trajectory replay inherited from earlier preflights. Single consumed prompt/one seed; no quality-equivalence/benchmark/hardware claim.'}
    (out/'independent_audit.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps(audit))


if __name__=='__main__':main()
