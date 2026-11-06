"""Complete, predeclared R1 cohorts with original and finite-mean controls."""
import argparse,json,math
from pathlib import Path
import numpy as np
from .r1_io_v1 import sha,source_guard
from .task_statistics_v1 import summarize

CONTROLS=['original','mean1','mean4','mean8','int2','int4','int8']


def safe_exp(x):
    return math.exp(x) if x<709 else None


def condition_summary(native,controls,margin,key):
    if len(native)!=5:raise ValueError('A complete5seedcohort required')
    sizes=np.asarray(controls['original']['unit_sizes'],np.int64)
    if any(a['unit_sizes']!=sizes.tolist() for a in [*native,*controls.values()]):raise ValueError('Dataunits changed')
    contrasts={}
    for arm in ['original','mean1']:
        baseline=np.asarray(controls[arm]['unit_metric_sums'],np.float64)
        gaps=np.stack([np.asarray(a['unit_metric_sums'],np.float64)-baseline for a in native])
        contrasts[arm]=summarize(gaps,sizes,margin,10000,key)
    means=[float(np.sum(a['unit_metric_sums'])/sizes.sum()) for a in native]
    return {'per_seed_mean_metric':means,'mean_metric':float(np.mean(means)),
        'paired_degradation':contrasts,'primary_quality_gate':all(v['quality_gate_met'] for v in contrasts.values()),
        'resources_by_seed':[a['resources'] for a in native]}


def load_case(p,protocol,slug):
    root=Path(p['output'])/'cases'/slug;rp=root/'results.json';ap=root/'independent_audit.json'
    r,a=json.loads(rp.read_text()),json.loads(ap.read_text())
    assert a['status']=='complete_R1_parameter_counts_resources_and_all_neural_batches_independently_replayed'
    assert a['source_sha256']==p['sources_sha256']['research/thrml_spw/broad/verify_r1_case_v1.py']
    assert not a['synthetic_only'] and a['protocol_sha256']==sha(protocol)
    assert a['results_sha256']==sha(rp) and a['case_spec']==p['cases'][slug]
    assert a['case']==slug and a['model_key']==p['cases'][slug]['model_key']
    assert a['model_point_sha256']==r['model_point_sha256']
    assert a['unique_parameter_scalars']==p['models'][a['model_key']]['unique_learned_scalars']
    return a,{str(rp):sha(rp),str(ap):sha(ap)}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--protocol',type=Path,required=True);ap.add_argument('--scope',choices=['primary','full'],required=True)
    args=ap.parse_args();p=json.loads(args.protocol.read_text());source_guard(p)
    assert p['status']=='complete_R1_execution_and_statistics_frozen_before_scoring'
    reports={};bindings={}
    for model,spec in p['models'].items():
        controls={}
        for arm in CONTROLS:
            controls[arm],b=load_case(p,args.protocol,model+'__'+arm);bindings.update(b)
        conditions={}
        cohorts=[(8192,1)] if args.scope=='primary' else [(c,k) for c in [1024,4096,8192] for k in [1,4,8]]
        for budget,bits in cohorts:
            seeds=spec['seeds_by_budget'][str(budget)];native=[]
            for seed in seeds:
                slug=f'{model}__K{bits}_C{budget}_seed{seed}';a,b=load_case(p,args.protocol,slug);bindings.update(b);native.append(a)
                assert a['case_spec']=={'model_key':model,'mode':'native','bits':bits,'budget':budget,'samples':budget//bits,'seed':seed}
            summary=condition_summary(native,controls,spec['margin'],spec['bootstrap_key'])
            summary.update(budget=budget,bits=bits,samples=budget//bits,seeds=seeds)
            assert all(r==summary['resources_by_seed'][0] for r in summary['resources_by_seed'])
            conditions[f'C{budget}_K{bits}']=summary
        controls_means={arm:float(np.sum(a['unit_metric_sums'])/np.sum(a['unit_sizes'])) for arm,a in controls.items()}
        reports[model]={'metric':controls['original']['metric'],'deterministic_control_mean_metrics':controls_means,
            'deterministic_control_count':1,'conditions':conditions,'primary_quality_gate':conditions['C8192_K1']['primary_quality_gate']}
    report={'status':'complete_predeclared_R1_cohort_statistics_pending_independent_audit','scope':args.scope,
        'protocol_sha256':sha(args.protocol),'sources_sha256':p['sources_sha256'],'inputs_sha256':bindings,
        'models':reports,'primary_both_models_pass':all(x['primary_quality_gate'] for x in reports.values()),
        'resource_unit':'Stochastic transitions per model reconstruction draw, reused across all evaluationexamples. Not per-input R2transitions. FiveMCdraws are independently evaluated, not ensembled.',
        'scope_note':'Whole-task R1 quality at frozen BF16 checkpoint points and finite controls, complete5seedcohorts only. Approximate intervals. No R2or physicalhardwareadvantageclaim.'}
    directory=Path(p['output'])/'statistics';directory.mkdir(exist_ok=True)
    with (directory/(args.scope+'.json')).open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps({'status':report['status'],'scope':args.scope,'primary_both_models_pass':report['primary_both_models_pass']}))

if __name__=='__main__':main()
