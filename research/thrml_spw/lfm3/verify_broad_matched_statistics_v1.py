"""Independent Decimal aggregation audit from original accepted case reports."""
from pathlib import Path
from decimal import Decimal, localcontext
import hashlib
import itertools
import json
import math


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    out=Path('results/thrml_spw/broad_verification_20261105/matched_statistics_v1')
    p=json.loads((out/'protocol.json').read_text());r=json.loads((out/'statistics.json').read_text())
    assert r['protocol_sha256']==sha(out/'protocol.json')
    assert p['independent_verifier_sha256']==sha(Path(__file__))
    assert p['source_sha256']==sha(Path(__file__).with_name('broad_matched_statistics_v1.py'))
    for name,digest in p['inputs_sha256'].items():assert sha(Path(name))==digest
    assert len(r['cohorts'])==3
    used=set();lowest={};weights=None
    with localcontext() as ctx:
        ctx.prec=70
        t=Decimal(str(p['critical']));margin=Decimal(str(p['quality_margin_NLL']))
        def close(actual,expected,tolerance='1e-12'):
            x=Decimal(str(actual));assert x.is_finite() and abs(x-expected)<Decimal(tolerance)
        def vector(record,values):
            assert len(record['values'])==len(values)==5
            for actual,expected in zip(record['values'],values):close(actual,expected)
            mean=sum(values)/5;se=(sum((v-mean)**2 for v in values)/20).sqrt()
            interval=[mean-t*se,mean+t*se]
            close(record['mean'],mean);close(record['standard_error'],se)
            for actual,expected in zip(record['MC95_t4_interval'],interval):close(actual,expected)
            return mean,interval
        for cohort,budget,first in zip(r['cohorts'],[1024,4096,8192],[202610350,202610420,202610600]):
            assert cohort['budget']==budget and cohort['seeds']==list(range(first,first+5))
            assert set(cohort['methods'])=={'K1','K4','K8'}
            native,controlled,costs={},{},{}
            for key,bits in [('K1',1),('K4',4),('K8',8)]:
                method=cohort['methods'][key];rows=cohort['costs_per_seed'][key]
                assert method['backend']==('GB10' if bits==1 or budget==1024 else 'H100')
                assert len(rows)==5
                audits=[]
                for seed in cohort['seeds']:
                    candidates=[]
                    for name in p['inputs_sha256']:
                        ap=Path(name)
                        if ap.parent.name!=str(seed) or ap.name not in ['independent_audit.json','h100_independent_audit_v3.json']:continue
                        a=json.loads(ap.read_text())
                        if (a.get('word_bits',1),a.get('budget',budget))==(bits,budget):candidates.append((ap,a))
                    assert len(candidates)==1,(budget,bits,seed)
                    ap,a=candidates[0];used.add(str(ap));rp=ap.with_name('results.json');used.add(str(rp))
                    assert str(rp) in p['inputs_sha256'] and a['results_sha256']==sha(rp)
                    original=json.loads(rp.read_text())
                    assert original['package_sha256']=='ed690a28d5976aee7d9932872edb30d92adfa26c4fa0e5f248834c3678cf6262'
                    assert a['targets']==2048 and a['unique_parameter_coordinates']==1728801536
                    if weights is None:weights=original['same_fit_BF16_weight_hashes']
                    assert original['same_fit_BF16_weight_hashes']==weights
                    row=rows[seed-first];assert row['seed']==seed
                    for field in ['retained_scheduled_evaluations','warmup_scheduled_evaluations','sampled_coordinates_across_uses','max_temporary_weight_count_bytes']:
                        assert row[field]==a[field]
                    assert row['logical_pbits_if_fully_resident']==bits*1728801536 and row['samples_per_pbit']==budget//bits
                    assert row['full_four_stage_case_wall_seconds']==original['elapsed_seconds']>0
                    assert math.isclose(row['retained_evaluations_per_full_case_wall_second'],a['retained_scheduled_evaluations']/original['elapsed_seconds'],rel_tol=1e-12)
                    if bits>1:
                        assert a['retained_scheduled_evaluations']==costs['K1'][seed-first]['retained_scheduled_evaluations']
                        assert a['warmup_scheduled_evaluations']==bits*costs['K1'][seed-first]['warmup_scheduled_evaluations']
                    audits.append(a)
                values=[Decimal(str(a['losses']['native']['mean_NLL_FP64'])) for a in audits]
                mean,_=vector(method['native_NLL'],values);close(method['perplexity_exp_mean_NLL'],mean.exp(),'1e-9')
                controls={name:Decimal(str(audits[0]['losses'][name]['mean_NLL_FP64'])) for name in ['BF16','same_fit_BF16','nominal_controls']}
                assert all({name:a['losses'][name]['mean_NLL_FP64'] for name in controls}==method['controls_NLL'] for a in audits)
                for name,value in controls.items():
                    gap,ci=vector(method['native_minus_own_control_NLL'][name],[v-value for v in values])
                    if name=='same_fit_BF16':
                        close(method['PPL_ratio_to_own_same_fit_BF16'],gap.exp())
                        for actual,expected in zip(method['PPL_ratio_MC95'],ci):close(actual,expected.exp())
                        assert method['conditional_2percent_gate_passed']==(ci[1]<=margin)
                close(method['own_same_fit_BF16_perplexity'],controls['same_fit_BF16'].exp(),'1e-9')
                if method['conditional_2percent_gate_passed'] and key not in lowest:
                    lowest[key]={'retained_work_budget':budget,'samples_per_pbit':budget//bits,'logical_pbits_per_parameter':bits,'backend':method['backend']}
                native[key]=values;controlled[key]=[v-controls['same_fit_BF16'] for v in values];costs[key]=rows
            for left,right in itertools.combinations(['K1','K4','K8'],2):
                c=cohort['paired_root_key_contrasts'][left+'_minus_'+right]
                vector(c['raw_native_NLL'],[a-b for a,b in zip(native[left],native[right])])
                vector(c['own_BF16_gap_difference'],[a-b for a,b in zip(controlled[left],controlled[right])])
                assert c['backends']==[cohort['methods'][left]['backend'],cohort['methods'][right]['backend']]
    assert used==set(p['inputs_sha256'])
    assert r['lowest_qualifying_budget_among_explored']==lowest
    audit={'status':'all45_case_inputs_Decimal_statistics_backend_labels_and_matched_work_independently_verified',
        'source_sha256':sha(Path(__file__)),'statistics_sha256':sha(out/'statistics.json'),
        'protocol_sha256':sha(out/'protocol.json'),'cases_verified':45,'unique_input_files':len(used),
        'scope':'Post-execution descriptive aggregation of accepted case audits. Hardware backends explicitly differ for higher-budget K1 versusK4/K8; no duplicate seed pooling, universal quality, or physical-cost claim.'}
    (out/'independent_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit))


if __name__=='__main__':main()
