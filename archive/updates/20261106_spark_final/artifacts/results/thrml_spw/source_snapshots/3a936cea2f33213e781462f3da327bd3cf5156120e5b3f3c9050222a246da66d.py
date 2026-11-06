"""Independent high-T cohort/binding audit using frequency-bootstrap oracle."""
from pathlib import Path
import argparse,json,math
import numpy as np
from .r1_io_v1 import sha,source_guard
from .verify_r1_statistics_v1 import verify_interval


def verify(p,protocol,report):
    source_guard(p);assert report['protocol_sha256']==sha(protocol) and report['sources_sha256']==p['sources_sha256']
    assert report['status']=='complete_highT_followup_statistics_pending_independent_audit'
    seeds=p['models']['gpt2']['seeds_by_budget']['262144'];spec=p['models']['gpt2']
    assert len(seeds)==len(set(seeds))==5 and spec['margin']==math.log(1.02)
    slugs=[f'gpt2__K1_C262144_seed{s}'for s in seeds];assert set(slugs)==set(p['cases'])
    parent_path=Path(p['control_protocol']);assert sha(parent_path)==p['control_protocol_sha256']
    parent=json.loads(parent_path.read_text());source_guard(parent)
    bindings={};all_audits=[]
    def read(slug,is_control):
        owner=parent if is_control else p;root=Path(owner['output'])/'cases'/slug
        a_path=root/'independent_audit.json';r_path=root/'results.json';a=json.loads(a_path.read_text());r=json.loads(r_path.read_text())
        assert a['status']=='complete_R1_parameter_counts_resources_and_all_neural_batches_independently_replayed'
        verifier='research/thrml_spw/broad/'+('verify_r1_case_v1.py'if is_control else'verify_highT_case_v1.py')
        assert a['source_sha256']==owner['sources_sha256'][verifier] and a['protocol_sha256']==sha(parent_path if is_control else protocol)
        assert a['case']==slug and a['case_spec']==owner['cases'][slug] and not a['synthetic_only']
        assert a['results_sha256']==sha(r_path) and a['model_point_sha256']==r['model_point_sha256']
        assert a['unique_parameter_scalars']==124439808 and a['canonical_parameters']==148
        assert a['model_key']=='gpt2' and a['evaluation_batches']==586 and a['metric']=='NLL'
        assert len(a['unit_sizes'])==62 and sum(a['unit_sizes'])==283225
        for path in [a_path,r_path]:
            h=sha(path);bindings[str(path)]=h
            if is_control:assert p['control_inputs_sha256'][str(path)]==h
        all_audits.append(a);return a
    controls={arm:read('gpt2__'+arm,True)for arm in ['original','mean1','mean4','mean8','int2','int4','int8']}
    native=[]
    for seed,slug in zip(seeds,slugs):
        assert p['cases'][slug]=={'model_key':'gpt2','mode':'native','bits':1,'budget':262144,'samples':262144,'seed':seed}
        native.append(read(slug,False))
    sizes=np.asarray(controls['original']['unit_sizes'],np.int64);assert all(a['unit_sizes']==sizes.tolist()for a in all_audits)
    summary=report['condition'];assert summary['bits']==1 and summary['samples']==summary['budget']==262144 and summary['seeds']==seeds
    means=[math.fsum(a['unit_metric_sums'])/int(sizes.sum())for a in native]
    np.testing.assert_allclose(summary['per_seed_mean_metric'],means,rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(summary['mean_metric'],math.fsum(means)/5,rtol=1e-12,atol=1e-12)
    for arm,a in controls.items():np.testing.assert_allclose(report['deterministic_control_mean_NLL'][arm],math.fsum(a['unit_metric_sums'])/int(sizes.sum()),rtol=1e-12,atol=1e-12)
    assert set(summary['paired_degradation'])=={'original','mean1'}
    for arm in ['original','mean1']:
        gaps=np.stack([np.asarray(a['unit_metric_sums'])-np.asarray(controls[arm]['unit_metric_sums'])for a in native])
        verify_interval(summary['paired_degradation'][arm],gaps,sizes,math.log(1.02),spec['bootstrap_key'])
    assert report['quality_gate_met']==summary['primary_quality_gate']==all(r['quality_gate_met']for r in summary['paired_degradation'].values())
    assert summary['resources_by_seed']==[a['resources']for a in native]
    expected={'logical_pbits':124439808,'logical_retained_device_evaluations':124439808*262144,
        'retained_device_evaluations':130744320*262144,'warmup_device_evaluations':130744320}
    assert all(a['resources']==expected for a in native)
    assert report['inputs_sha256']==bindings
    return True


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--protocol',type=Path,required=True);args=ap.parse_args()
    p=json.loads(args.protocol.read_text());path=Path(p['output'])/'statistics/followup.json';r=json.loads(path.read_text());verify(p,args.protocol,r)
    audit={'status':'highT_followup_full_fixed_cohort_statistics_and_all12case_bindings_independently_verified',
        'source_sha256':sha(__file__),'protocol_sha256':sha(args.protocol),'statistics_sha256':sha(path),
        'cases_checked':12,'quality_gate_met':r['quality_gate_met'],
        'scope':'Five new fixedMCdraws on already-exposedtext, against immutable original/finite-mean controls. Original8192primary remainsunchanged. Not universal numericalBF16equivalence or physicalhardwareproof.'}
    with path.with_name('followup_audit.json').open('x')as f:json.dump(audit,f,indent=2);f.write('\n')
    print(json.dumps(audit))

if __name__=='__main__':main()
