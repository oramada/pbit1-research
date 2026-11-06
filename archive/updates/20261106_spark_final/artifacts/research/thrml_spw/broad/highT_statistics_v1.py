"""Complete five-seed GPT2 high-T follow-up; original primary stays unchanged."""
from pathlib import Path
import argparse,json,math
import numpy as np
from .r1_io_v1 import sha,source_guard
from .r1_statistics_v1 import condition_summary,CONTROLS


def cohort(p):
    seeds=p['models']['gpt2']['seeds_by_budget']['262144']
    if len(seeds)!=5 or len(set(seeds))!=5:raise ValueError('Complete five unique fixed seeds required')
    expected={f'gpt2__K1_C262144_seed{s}':{'model_key':'gpt2','mode':'native','bits':1,'budget':262144,'samples':262144,'seed':s}for s in seeds}
    if p['cases']!=expected:raise ValueError('Exact frozen five-case cohort required')
    return list(expected)


def evidence(p,protocol,slug,control=False):
    if control:
        parent=Path(p['control_protocol']);assert sha(parent)==p['control_protocol_sha256']
        parent_p=json.loads(parent.read_text());source_guard(parent_p)
        root=Path(parent_p['output'])/'cases'/slug;expected=parent_p['cases'][slug]
        expected_auditor=parent_p['sources_sha256']['research/thrml_spw/broad/verify_r1_case_v1.py']
        expected_protocol=sha(parent)
    else:
        root=Path(p['output'])/'cases'/slug;expected=p['cases'][slug]
        expected_auditor=p['sources_sha256']['research/thrml_spw/broad/verify_highT_case_v1.py'];expected_protocol=sha(protocol)
    rp,ap=root/'results.json',root/'independent_audit.json';r,a=json.loads(rp.read_text()),json.loads(ap.read_text())
    assert a['status']=='complete_R1_parameter_counts_resources_and_all_neural_batches_independently_replayed'
    assert a['results_sha256']==sha(rp) and a['source_sha256']==expected_auditor and a['protocol_sha256']==expected_protocol
    assert a['case_spec']==expected and a['case']==slug and a['model_key']=='gpt2' and not a['synthetic_only']
    assert a['unique_parameter_scalars']==124439808 and a['model_point_sha256']==r['model_point_sha256']
    assert a['evaluation_batches']==586 and a['metric']=='NLL'
    assert len(a['unit_sizes'])==62 and sum(a['unit_sizes'])==283225
    bindings={str(rp):sha(rp),str(ap):sha(ap)}
    if control:
        for path,h in bindings.items():assert p['control_inputs_sha256'][path]==h
    return a,bindings


def build(p,protocol):
    source_guard(p);slugs=cohort(p);bindings={};controls={};spec=p['models']['gpt2']
    assert spec['margin']==math.log(1.02) and p['sampling_budget']==262144 and p['logical_pbits_per_scalar']==1
    for arm in CONTROLS:
        controls[arm],b=evidence(p,protocol,'gpt2__'+arm,True);bindings.update(b)
    native=[]
    for slug in slugs:
        a,b=evidence(p,protocol,slug);bindings.update(b);native.append(a)
    summary=condition_summary(native,controls,spec['margin'],spec['bootstrap_key'])
    summary.update(bits=1,budget=262144,samples=262144,seeds=spec['seeds_by_budget']['262144'])
    r=native[0]['resources'];assert all(a['resources']==r for a in native)
    assert r['logical_pbits']==124439808 and r['logical_retained_device_evaluations']==124439808*262144
    assert r['retained_device_evaluations']==130744320*262144 and r['warmup_device_evaluations']==130744320
    return {'status':'complete_highT_followup_statistics_pending_independent_audit','protocol_sha256':sha(protocol),
        'sources_sha256':p['sources_sha256'],'inputs_sha256':bindings,'condition':summary,
        'deterministic_control_mean_NLL':{arm:float(np.sum(a['unit_metric_sums'])/np.sum(a['unit_sizes']))for arm,a in controls.items()},
        'quality_gate_met':summary['primary_quality_gate'],
        'resource_unit':'One reconstructedmodel draw reused across all283225targets; five draws separately evaluated, not ensembled.',
        'scope':'Distinct post-primary follow-up on already-exposed WikiText2text with five prospectively fixed new stochastic keys. Original8192primary miss remainsunchanged. ConditionalMC and crossedarticle/seed approximate intervals; not new-text replication, numericalBF16equivalence or hardwareadvantage.'}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--protocol',type=Path,required=True);args=ap.parse_args()
    p=json.loads(args.protocol.read_text());r=build(p,args.protocol);out=Path(p['output'])/'statistics';out.mkdir(exist_ok=True)
    with (out/'followup.json').open('x')as f:json.dump(r,f,indent=2);f.write('\n')
    print(json.dumps({'status':r['status'],'quality_gate_met':r['quality_gate_met']}))

if __name__=='__main__':main()
