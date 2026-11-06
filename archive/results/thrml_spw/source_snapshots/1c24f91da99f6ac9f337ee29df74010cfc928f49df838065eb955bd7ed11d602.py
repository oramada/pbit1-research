"""Independent loss, unique coverage, settings and zero-sampling control audit."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    root=Path('results/thrml_spw/lfm3_1p7b');out=root/'precision_floor'
    r=json.loads((out/'results.json').read_text());p=json.loads((out/'protocol.json').read_text())
    inv=json.loads((out/'conversion_inventory.json').read_text())
    canonical=json.loads((root/'meta_inventory.json').read_text())
    expected={x['name']:x for x in canonical['parameters']}
    modules={x['name']:x for x in inv['modules']};seen=set()
    for m in modules.values():
        key=('model.embed_tokens' if m['shared_coordinate_group']=='token_embedding_head' else m['name'])+'.weight'
        assert key in expected and m['shape']==expected[key]['shape'];seen.add(key)
    assert seen==set(expected) and sum(x['numel'] for x in expected.values())==r['total_parameters']==r['unique_parameters_converted']==1728801536
    for file,key in [('precision_floor.py','source_sha256'),('precision_worker.py','worker_sha256')]:assert sha(Path(__file__).with_name(file))==p[key]
    assert sha(out/'outputs.npz')==r['outputs_sha256']
    a=np.load(out/'outputs.npz');prior=np.load(root/'group128_T256/outputs.npz')
    np.testing.assert_array_equal(a['BF16'],prior['BF16']);np.testing.assert_array_equal(a['identity_bridge'],a['BF16'])
    np.testing.assert_array_equal(a['mean_f8_a8'],prior['nominal_controls'])
    ids=a['input_ids'];metrics=[]
    assert len(r['metrics'])==5
    for metric in r['metrics']:
        y=a[metric['stage']];assert np.isfinite(y).all() and y.shape==a['BF16'].shape
        ce=F.cross_entropy(torch.from_numpy(y[:,:-1].astype(np.float64).reshape(-1,y.shape[-1])),torch.from_numpy(ids[:,1:].reshape(-1))).item()
        np.testing.assert_allclose(metric['diagnostic_CE_FP64'],ce,rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(metric['logit_MSE_vs_BF16'],np.mean((y.astype(np.float64)-a['BF16'])**2),rtol=1e-12,atol=1e-12)
        assert metric['argmax_agreement']==float(np.mean(y.argmax(-1)==a['BF16'].argmax(-1)))
        metrics.append({'stage':metric['stage'],'diagnostic_CE_FP64':ce})
    for stage in p['stages'][1:]:
        _,fb,ab=stage.split('_');fb=int(fb[1:]);ab=int(ab[1:])
        calls=[c for c in r['operator_calls'] if c['stage']==stage]
        assert len(calls)==len(modules) and {c['module'] for c in calls}==set(modules)
        for c in calls:
            assert c['field_bits']==fb and c['activation_bits']==ab and c['finite']
            assert c['retained_evaluations']==c['warmup_evaluations']==0
            m=modules[c['module']];shape=m['shape']
            count=len(np.unique(ids))*shape[1] if m['kind']=='embedding' else int(np.prod(shape))
            assert c['weight_coordinates']==count
    groups=json.loads((root/'range_moments/results.json').read_text())['summary']['block128']['groups']
    n=1728801536
    payload={str(bits):{'packed_commands_bytes':(n*bits+7)//8,
        'packed_commands_plus_state_plus_FP32_metadata_bytes':(n*bits+7)//8+(n+7)//8+groups*8}
        for bits in [8,12]}
    audit={'status':'all_four_full_parameter_mean_controls_verified','source_sha256':sha(Path(__file__)),
        'results_sha256':sha(out/'results.json'),'unique_parameters':n,'coverage_fraction':1.,
        'operator_modules_per_condition':len(modules),'mean_operator_calls_verified':len(r['operator_calls']),
        'prior_8bit_field_ACT8_control_exact':True,'native_sampling_evaluations':0,
        'metrics':metrics,'logical_packed_payload_assumptions':payload,
        'scope':'Deterministic nominal operator-mean controls for one consumed16-token prompt. Not expected full stochastic model quality, benchmark or equivalence test; actual code storage/frame overhead not included in packed payload.'}
    (out/'independent_audit.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps(audit))


if __name__=='__main__':main()
