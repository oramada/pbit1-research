"""Independent count-weighted bootstrap/MC audit; no reserved token access."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from .holdout_batch_layout import assemble_losses


STAGES=['BF16','same_fit_BF16','nominal_controls','native']


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def close(actual,expected):
    a,b=np.asarray(actual,dtype=np.float64),np.asarray(expected,dtype=np.float64)
    assert np.isfinite(a).all() and np.isfinite(b).all()
    np.testing.assert_allclose(a,b,rtol=2e-12,atol=2e-12,equal_nan=False)


def verify_summary(report,sets,parameters):
    assert set(sets)==set(STAGES)
    arrays={}
    for stage in STAGES:
        s,l=[np.asarray(v,dtype=np.float64) for v in sets[stage]]
        assert s.shape==(5,96,64) and l.shape==(5,16,256)
        assert np.isfinite(s).all() and np.isfinite(l).all()
        if stage!='native':assert np.all(s==s[0]) and np.all(l==l[0])
        arrays[stage]=s,l
        means=[(math.fsum(s[i].ravel())+math.fsum(l[i].ravel()))/10240 for i in range(5)]
        record=report['absolute_NLL'][stage]
        close(record['per_seed_NLL'],means);close(record['mean_NLL'],math.fsum(means)/5)
        close(record['PPL_exp_mean_NLL'],math.exp(math.fsum(means)/5))
        close(record['short_mean_NLL'],math.fsum(s.ravel())/s.size)
        close(record['long_mean_NLL'],math.fsum(l.ravel())/l.size)
    n=parameters['bootstrap_replicates'];rng=np.random.default_rng(parameters['bootstrap_key'])
    ss=rng.integers(0,5,(n,5));aa=rng.integers(0,24,(n,24));bb=rng.integers(0,16,(n,16))
    sc=np.stack([np.bincount(v,minlength=5) for v in ss]).astype(float)
    ac=np.stack([np.bincount(v,minlength=24) for v in aa]).astype(float)
    bc=np.stack([np.bincount(v,minlength=16) for v in bb]).astype(float)
    gate={}
    for control in STAGES[:-1]:
        short=arrays['native'][0]-arrays[control][0]
        long=arrays['native'][1]-arrays[control][1]
        sa=np.array([[math.fsum(short[i,4*j:4*j+4].ravel())/256 for j in range(24)] for i in range(5)])
        la=np.array([[math.fsum(long[i,j])/256 for j in range(16)] for i in range(5)])
        # Product of sampled seed frequencies and sampled article frequencies;
        # independent of the producer's expanded-index bootstrap implementation.
        sd=np.einsum('rs,sa,ra->r',sc,sa,ac)/120
        ld=np.einsum('rs,sa,ra->r',sc,la,bc)/80
        sv=sa.mean(1);lv=la.mean(1)
        summary=report['paired_contrasts'][control]
        assert summary['bootstrap_replicates']==n and summary['bootstrap_key']==parameters['bootstrap_key']
        assert summary['articles_short_long']==[24,16] and summary['targets_per_article']==256
        close(summary['pooled_target_weights'],[.6,.4])
        for name,values,draws in [('short',sv,sd),('long',lv,ld),('pooled',.6*sv+.4*lv,.6*sd+.4*ld)]:
            rec=summary['contrasts'][name];mean=math.fsum(values)/5
            se=math.sqrt(math.fsum((float(v)-mean)**2 for v in values)/4/5)
            interval=[mean-2.7764451051977987*se,mean+2.7764451051977987*se]
            bootstrap=np.quantile(draws,[.025,.975],method='linear')
            close(rec['per_seed_mean_NLL_gap'],values);close(rec['mean_NLL_gap'],mean)
            close(rec['MC_standard_error_conditional_on_articles'],se)
            close(rec['MC95_t4_interval_conditional_on_articles'],interval)
            close(rec['stratified_crossed_bootstrap_percentile95_interval'],bootstrap)
            close(rec['PPL_ratio_exp_mean_gap'],math.exp(mean))
            close(rec['bootstrap_PPL_ratio_interval'],np.exp(bootstrap))
            if control=='same_fit_BF16':
                # Decision must follow the independently checked reported bounds.
                gate[name]={'conditional_MC_upper_within_margin':rec['MC95_t4_interval_conditional_on_articles'][1]<=parameters['NLL_margin'],
                    'crossed_bootstrap_upper_within_margin':rec['stratified_crossed_bootstrap_percentile95_interval'][1]<=parameters['NLL_margin']}
    assert report['primary_same_fit_BF16_tolerance_checks']==gate
    assert type(report['quality_tolerance_met_in_both_strata_and_pooled']) is bool
    assert report['quality_tolerance_met_in_both_strata_and_pooled']==all(all(v.values()) for v in gate.values())
    close(report['NLL_margin'],parameters['NLL_margin'])
    return True


def main():
    root=Path('results/thrml_spw/lfm3_1p7b');out=root/'holdout_native'
    pp=root/'holdout_native_protocol.json';ap=root/'holdout_native_statistics_protocol.json'
    p,analysis=json.loads(pp.read_text()),json.loads(ap.read_text())
    report=json.loads((out/'statistics.json').read_text())
    assert p['status']=='complete_execution_and_statistics_protocol_frozen_before_scoring'
    assert analysis['status']=='complete_paired_holdout_analysis_frozen_before_scoring'
    assert sha(ap)==p['statistics_protocol_sha256']==report['analysis_protocol_sha256']
    assert sha(pp)==report['execution_protocol_sha256']
    assert report['sources_sha256']==analysis['sources_sha256']
    assert analysis['sources_sha256'][Path(__file__).name]==sha(Path(__file__))
    for name,digest in analysis['sources_sha256'].items():assert sha(Path(__file__).with_name(name))==digest
    assert report['status']=='complete_five_seed_reserved_statistics_produced_pending_independent_audit'
    assert report['seeds']==p['seeds'] and len(p['seeds'])==len(set(p['seeds']))==5
    assert report['samples']==p['samples'] and report['pbits_per_parameter']==1
    assert report['unique_parameter_coordinates']==1728801536 and report['targets_per_seed']==10240 and report['articles']==40
    sets={stage:[[],[]] for stage in STAGES};resources=[]
    expected_bindings={};expected_cases=[]
    for seed in p['seeds']:
        rp,case_ap=out/str(seed)/'results.json',out/str(seed)/'independent_audit.json'
        r,a=json.loads(rp.read_text()),json.loads(case_ap.read_text())
        assert a['status']=='full_reserved_case_raw_losses_controls_coverage_and_resources_independently_verified'
        assert a['source_sha256']==p['independent_case_verifier_sha256'] and a['results_sha256']==sha(rp)
        assert a['shared_protocol_sha256']==sha(pp) and a['seed']==seed
        for stage in STAGES:
            records=[{'batch_index':x['batch_index'],'token_NLL':x['token_NLL']} for x in r['metrics'] if x['stage']==stage]
            s,l=assemble_losses(p['batch_plan'],records)
            close((s.sum()+l.sum())/10240,a['losses'][stage]['mean_NLL_FP64'])
            sets[stage][0].append(s);sets[stage][1].append(l)
        resources.append({key:a[key] for key in ['retained_scheduled_evaluations','warmup_scheduled_evaluations',
            'sampled_coordinates_across_uses','max_temporary_weight_count_bytes','max_simultaneous_sampler_state_slots']})
        expected_bindings[str(rp)]=sha(rp);expected_bindings[str(case_ap)]=sha(case_ap)
        expected_cases.append({'seed':seed,'results_sha256':sha(rp),'audit_sha256':sha(case_ap)})
    assert report['inputs_sha256']==expected_bindings and report['resources_per_seed']==resources
    assert report['cases']==expected_cases
    assert all(v==resources[0] for v in resources)
    verify_summary(report,{stage:tuple(np.stack(v) for v in arrays) for stage,arrays in sets.items()},analysis['parameters'])
    audit={'status':'complete_five_seed_article_MC_statistics_and_all_resource_bindings_independently_verified',
        'source_sha256':sha(Path(__file__)),'statistics_sha256':sha(out/'statistics.json'),
        'execution_protocol_sha256':sha(pp),'analysis_protocol_sha256':sha(ap),
        'seeds':p['seeds'],'samples':p['samples'],'targets_per_seed':10240,'articles':40,
        'quality_tolerance_met_in_both_strata_and_pooled':report['quality_tolerance_met_in_both_strata_and_pooled'],
        'scope':'Independent factorized bootstrap-frequency and MC arithmetic, complete fixed five-case bindings and resources. Approximate intervals, no formal coverage guarantee, numerical BF16 equivalence or hardware advantage claim.'}
    with (out/'statistics_audit.json').open('x') as f:json.dump(audit,f,indent=2)
    print(json.dumps(audit))


if __name__=='__main__':main()
