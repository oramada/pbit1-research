"""Independent saved-output/coverage audit of the end-to-end LFM3 bridge smoke."""
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    out=Path('results/thrml_spw/lfm3_1p7b/bridge_smoke')
    result=json.loads((out/'results.json').read_text())
    protocol=json.loads((out/'protocol.json').read_text())
    inventory=json.loads((out/'conversion_inventory.json').read_text())
    assert sha(out/'outputs.npz')==result['outputs_sha256']
    raw=np.load(out/'outputs.npz');ids=raw['input_ids']
    np.testing.assert_array_equal(raw['BF16'],raw['identity_bridge'])
    expected={r['name']:r for r in inventory['modules']}
    count=sum(int(np.prod(r['shape'])) for r in inventory['modules'])
    assert count==inventory['converted_unique_parameters']==result['converted_unique_parameters']
    metrics=0
    for row in result['metrics']:
        logits=raw[row['stage']];baseline=raw['BF16']
        assert logits.shape==baseline.shape and np.isfinite(logits).all()
        z=logits[:,:-1,:].astype(np.float64);mx=z.max(-1)
        lognorm=mx+np.log(np.exp(z-mx[...,None]).sum(-1))
        target=np.take_along_axis(z,ids[:,1:,None],axis=-1)[...,0]
        ce=np.mean(lognorm-target)
        np.testing.assert_allclose(row['diagnostic_next_token_CE'],ce,rtol=2e-6,atol=2e-6)
        np.testing.assert_allclose(row['logit_MSE_vs_BF16'],np.mean((logits.astype(np.float64)-baseline)**2),rtol=1e-12,atol=1e-12)
        assert row['argmax_agreement']==float(np.mean(logits.argmax(-1)==baseline.argmax(-1)))
        metrics+=3
    native_work=0;warmup=0
    for stage in ['nominal_controls','native']:
        calls=[c for c in result['operator_calls'] if c['stage']==stage]
        assert len(calls)==len(expected)
        assert {c['module'] for c in calls}==set(expected)
        for c in calls:
            shape=expected[c['module']]['shape'];n=int(np.prod(shape))
            assert c['finite'] and c['weight_coordinates']==n
            if stage=='native':
                assert c['retained_evaluations']==16*(n+c['padding_nodes'])
                assert c['warmup_evaluations']==n+c['padding_nodes']
                native_work+=c['retained_evaluations'];warmup+=c['warmup_evaluations']
            else:assert c['retained_evaluations']==0
    audit={'status':'saved_logits_metrics_and_conversion_coverage_verified',
           'source_sha256':sha(Path(__file__)),'results_sha256':sha(out/'results.json'),
           'metrics_recomputed':metrics,'modules_per_stochastic_forward':len(expected),
           'converted_unique_parameters':count,'converted_fraction':count/inventory['total_unique_parameters'],
           'native_retained_evaluations':native_work,'native_warmup_evaluations':warmup,
           'scope':'Saved-logit metric and coverage audit only; native trajectory correctness inherited from prior exact operator replay tests. Single short consumed prompt/one seed; no benchmark or all-parameter claim.'}
    (out/'independent_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit))


if __name__=='__main__':main()
