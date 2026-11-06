"""Audit full unique parameter coverage, saved losses and node-work receipts."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,required=True);args=parser.parse_args()
    assert args.seed in range(202610300,202610305)
    torch.set_num_threads(4)
    root=Path('results/thrml_spw/lfm3_1p7b');out=root/'language_native_v2'/str(args.seed)
    protocol=json.loads((out/'protocol.json').read_text());r=json.loads((out/'results.json').read_text())
    inventory=json.loads((out/'conversion_inventory.json').read_text())
    meta=json.loads((root/'meta_inventory.json').read_text())
    for source,key in [('language_native_v2.py','source_sha256'),('precision_worker.py','worker_sha256')]:
        assert sha(Path(__file__).with_name(source))==protocol[key]
    assert sha(out/'outputs.npz')==r['outputs_sha256']
    assert sha(root/'language_subset/tokens.npz')==protocol['input_sha256']
    assert sha(root/'language_native_v2_protocol.json')==protocol['cohort_protocol_sha256']
    assert protocol['seed']==args.seed and protocol['samples']==1024
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
    np.testing.assert_array_equal(ids,np.load(root/'language_subset/tokens.npz')['tokens'])
    assert ids.shape==(8,65)
    np.testing.assert_array_equal(baseline[:,:-1],np.load(root/'language_controls_batch/BF16_logits.npy'))
    np.testing.assert_array_equal(baseline,a['identity_bridge'])
    losses=[]
    for metric in r['metrics']:
        y=a[metric['stage']];assert np.isfinite(y).all() and y.shape==baseline.shape
        # Independent Torch double log-softmax, versus producer NumPy logsumexp.
        nlls=[]
        for window in range(8):
            nll=F.cross_entropy(torch.from_numpy(y[window,:-1].astype(np.float64)),torch.from_numpy(ids[window,1:]),reduction='none').numpy()
            np.testing.assert_allclose(nll,metric['token_NLL'][window],rtol=1e-12,atol=1e-12)
            nlls.extend(nll.tolist())
        ce=float(np.mean(nlls))
        np.testing.assert_allclose(metric['mean_NLL_FP64'],ce,rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(metric['subset_perplexity'],np.exp(ce),rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(metric['logit_MSE_vs_BF16'],np.mean((y.astype(np.float64)-baseline)**2),rtol=1e-12,atol=1e-12)
        assert metric['argmax_agreement']==float(np.mean(y.argmax(-1)==baseline.argmax(-1)))
        losses.append({'stage':metric['stage'],'mean_NLL_FP64':ce,'subset_perplexity':float(np.exp(ce))})
    retained=0;warmup=0;drawn=0
    for stage in ['nominal_controls','native']:
        calls=[c for c in r['operator_calls'] if c['stage']==stage]
        assert len(calls)==len(modules) and {c['module'] for c in calls}==set(modules)
        for c in calls:
            m=modules[c['module']];kind=m['kind'];shape=m['shape']
            assert c['kind']==kind and c['finite']
            assert c['field_bits']==12 and c['activation_bits']==12
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
    assert protocol['samples']==1024
    previous_controls_equal=None
    if args.seed>202610300:
        prior=np.load(root/'language_native_v2/202610300/outputs.npz')
        for name in ['BF16','identity_bridge','nominal_controls']:np.testing.assert_array_equal(a[name],prior[name])
        previous_controls_equal=True
    audit={'matched_first_seed_deterministic_controls':previous_controls_equal,'matched_BF16_controls_exact':True,'samples':protocol['samples'],'status':'all_unique_parameters_and_saved_metrics_verified','source_sha256':sha(Path(__file__)),
        'results_sha256':sha(out/'results.json'),'unique_parameters_converted':total,'coverage_fraction':1.0,
        'operator_modules_per_forward':len(modules),'sampled_coordinates_across_uses':drawn,
        'retained_evaluations_including_padding':retained,'warmup_evaluations_including_padding':warmup,
        'losses':losses,'seed':args.seed,'scope':'Full unique parameter coverage and all512 saved token losses verified. BF16 matches frozen language controls exactly; later seeds match first seed deterministic controls. Native trajectories validated on earlier operator preflights, not replayed at full model scale. Five-seed exploratory subset; no equivalence or hardware claim.'}
    with (out/'independent_audit.json').open('x') as f:json.dump(audit,f,indent=2)
    print(json.dumps(audit))


if __name__=='__main__':main()
