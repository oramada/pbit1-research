"""Bounded-memory frequency bootstrap oracle and complete evidence bindings."""
import argparse,json,math
from pathlib import Path
import numpy as np
from .r1_io_v1 import sha,source_guard


def verify_interval(report,gaps,sizes,margin,key):
    gaps=np.asarray(gaps,np.float64);sizes=np.asarray(sizes,np.int64)
    means=np.array([math.fsum(v)/int(sizes.sum()) for v in gaps]);mean=math.fsum(means)/5
    se=math.sqrt(math.fsum((float(x)-mean)**2 for x in means)/20)
    mc=[mean-2.7764451051977987*se,mean+2.7764451051977987*se]
    n=10000;rng=np.random.default_rng(key);seeds=rng.integers(0,5,(n,5));draws=np.empty(n)
    for begin in range(0,n,64):
        end=min(begin+64,n);units=rng.integers(0,len(sizes),(end-begin,len(sizes)))
        sc=np.stack([np.bincount(v,minlength=5) for v in seeds[begin:end]])
        uc=np.stack([np.bincount(v,minlength=len(sizes)) for v in units])
        draws[begin:end]=np.einsum('rs,su,ru->r',sc,gaps,uc)/(5*(uc@sizes))
    boot=np.quantile(draws,[.025,.975],method='linear')
    for a,b in [(report['per_seed_mean_gap'],means),(report['mean_gap'],mean),(report['MC_standard_error'],se),
                (report['MC95_t4_interval_conditional_on_data'],mc),
                (report['crossed_seed_data_bootstrap_percentile95_interval'],boot)]:
        np.testing.assert_allclose(a,b,rtol=1e-12,atol=1e-12)
    assert report['data_units']==len(sizes) and report['scored_observations_per_seed']==int(sizes.sum())
    assert report['bootstrap_key']==key and report['bootstrap_replicates']==n and report['margin']==margin
    # The independently checked reported bounds define the exact floating decision.
    assert report['MC_upper_within_margin']==(report['MC95_t4_interval_conditional_on_data'][1]<=margin)
    assert report['bootstrap_upper_within_margin']==(report['crossed_seed_data_bootstrap_percentile95_interval'][1]<=margin)
    assert report['quality_gate_met']==(report['MC_upper_within_margin'] and report['bootstrap_upper_within_margin'])
    return True


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--protocol',type=Path,required=True);ap.add_argument('--scope',choices=['primary','full'],required=True)
    args=ap.parse_args();p=json.loads(args.protocol.read_text());source_guard(p)
    directory=Path(p['output'])/'statistics';path=directory/(args.scope+'.json');r=json.loads(path.read_text())
    assert r['protocol_sha256']==sha(args.protocol) and r['sources_sha256']==p['sources_sha256'] and r['scope']==args.scope
    assert r['status']=='complete_predeclared_R1_cohort_statistics_pending_independent_audit'
    assert set(r['models'])==set(p['models']);bindings={};case_cache={}
    def read(slug):
        if slug in case_cache:return case_cache[slug]
        root=Path(p['output'])/'cases'/slug;rp=root/'results.json';ap=root/'independent_audit.json'
        a=json.loads(ap.read_text());produced=json.loads(rp.read_text())
        assert a['status']=='complete_R1_parameter_counts_resources_and_all_neural_batches_independently_replayed'
        assert a['source_sha256']==p['sources_sha256']['research/thrml_spw/broad/verify_r1_case_v1.py']
        assert a['results_sha256']==sha(rp) and a['protocol_sha256']==sha(args.protocol) and not a['synthetic_only']
        assert a['case_spec']==p['cases'][slug] and a['case']==slug
        assert a['model_point_sha256']==produced['model_point_sha256']
        assert a['unique_parameter_scalars']==p['models'][a['model_key']]['unique_learned_scalars']
        bindings[str(ap)]=sha(ap);bindings[str(rp)]=sha(rp);case_cache[slug]=a;return a
    primary=[]
    for model,spec in p['models'].items():
        controls={arm:read(model+'__'+arm) for arm in ['original','mean1','mean4','mean8','int2','int4','int8']}
        record=r['models'][model]
        assert record['deterministic_control_count']==1 and record['metric']==controls['original']['metric']
        sizes=np.asarray(controls['original']['unit_sizes'],np.int64)
        for arm,a in controls.items():
            assert a['unit_sizes']==sizes.tolist()
            expected=math.fsum(a['unit_metric_sums'])/int(sizes.sum())
            np.testing.assert_allclose(record['deterministic_control_mean_metrics'][arm],expected,rtol=1e-12,atol=1e-12)
        pairs=[(8192,1)] if args.scope=='primary' else [(c,k) for c in [1024,4096,8192] for k in [1,4,8]]
        assert set(record['conditions'])=={f'C{c}_K{k}' for c,k in pairs}
        for c,k in pairs:
            summary=record['conditions'][f'C{c}_K{k}'];seeds=spec['seeds_by_budget'][str(c)]
            assert summary['bits']==k and summary['budget']==c and summary['samples']==c//k and summary['seeds']==seeds
            native=[read(f'{model}__K{k}_C{c}_seed{seed}') for seed in seeds]
            assert len(native)==len(seeds)==len(set(seeds))==5 and all(a['unit_sizes']==sizes.tolist() for a in native)
            means=[math.fsum(a['unit_metric_sums'])/int(sizes.sum()) for a in native]
            np.testing.assert_allclose(summary['per_seed_mean_metric'],means,rtol=1e-12,atol=1e-12)
            np.testing.assert_allclose(summary['mean_metric'],math.fsum(means)/5,rtol=1e-12,atol=1e-12)
            assert summary['resources_by_seed']==[a['resources'] for a in native]
            assert all(a['resources']==native[0]['resources'] for a in native)
            assert set(summary['paired_degradation'])=={'original','mean1'}
            for arm in ['original','mean1']:
                gaps=np.stack([np.asarray(a['unit_metric_sums'])-np.asarray(controls[arm]['unit_metric_sums']) for a in native])
                verify_interval(summary['paired_degradation'][arm],gaps,sizes,spec['margin'],spec['bootstrap_key'])
            assert summary['primary_quality_gate']==all(x['quality_gate_met'] for x in summary['paired_degradation'].values())
        assert record['primary_quality_gate']==record['conditions']['C8192_K1']['primary_quality_gate'];primary.append(record['primary_quality_gate'])
    assert r['inputs_sha256']==bindings and r['primary_both_models_pass']==all(primary)
    audit={'status':'all_predeclared_R1_task_statistics_and_case_bindings_independently_verified','scope':args.scope,
        'source_sha256':sha(__file__),'statistics_sha256':sha(path),'protocol_sha256':sha(args.protocol),
        'cases_checked':len(case_cache),'primary_both_models_pass':r['primary_both_models_pass'],
        'scope_note':'Independent bounded-memory frequency-weighted bootstrap and MC arithmetic, completefixedcohorts and audited neuralreplaybindings. Not universal BF16 numericalequivalence or physicalhardwareproof.'}
    with (directory/(args.scope+'_audit.json')).open('x') as f:json.dump(audit,f,indent=2);f.write('\n')
    print(json.dumps(audit))

if __name__=='__main__':main()
