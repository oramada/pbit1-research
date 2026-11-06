"""Post-execution, backend-labelled aggregation of completed matched-work cases."""
from pathlib import Path
import hashlib
import itertools
import json
import math

ROOT = Path('results/thrml_spw/lfm3_1p7b')
OUT = Path('results/thrml_spw/broad_verification_20261105/matched_statistics_v1')
MIRROR = Path('results/thrml_spw/broad_verification_20261105/h100_mirror') / ROOT / 'modal_h100_word_v2'
CRITICAL = 2.7764451051977987
MARGIN = math.log(1.02)


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def summary(values):
    assert len(values) == 5 and all(math.isfinite(v) for v in values)
    mean = math.fsum(values)/5
    se = math.sqrt(math.fsum((v-mean)**2 for v in values)/20)
    return {'values': values, 'mean': mean, 'standard_error': se,
        'MC95_t4_interval': [mean-CRITICAL*se, mean+CRITICAL*se]}


def main():
    OUT.mkdir(exist_ok=False)
    inputs, cohorts, minimum = {}, [], {}
    for budget, first in [(1024,202610350),(4096,202610420),(8192,202610600)]:
        seeds = list(range(first,first+5)); methods={}; costs={}; cases={}
        for bits in [1,4,8]:
            native, controls, rows, records = [], None, [], []
            backend = 'GB10' if bits==1 or budget==1024 else 'H100'
            for seed in seeds:
                if bits==1:
                    name={1024:'counted_mean_validation',4096:'counted_mean_t4096',8192:'counted_mean_t8192'}[budget]
                    folder=ROOT/name/'mean'/str(seed); filename='independent_audit.json'
                    expected={1024:'a3b223b1b9a7cc27cb525caaa918ab75decdadc331cca2ea7be56d2c0195e372',
                        4096:'487293960fdb834ede2a4de166f145767c4193f5430d49c133db9656d996a070',
                        8192:'f85318c765d6a6177c310c757cf0714c1cac0f849676dbb2e06d7c30da01a85d'}[budget]
                else:
                    folder=(ROOT/'adapted_word_validation_v2' if budget==1024 else MIRROR)/f'C{budget}'/f'K{bits}'/'mean'/str(seed)
                    filename='independent_audit.json' if backend=='GB10' else 'h100_independent_audit_v3.json'
                    expected='e1e326b551691fbeacc2a774aaa58ec6d499fb896b00fb1dcc1943489a82f3f8' if backend=='GB10' else '3a8232526459d8b299c6c785903137a3cf834ae7fce6724dcf85c7f716502303'
                ap,rp=folder/filename,folder/'results.json';a=json.loads(ap.read_text());r=json.loads(rp.read_text())
                assert a['source_sha256']==expected and a['results_sha256']==sha(rp)
                assert a['seed']==seed and a['targets']==2048 and a['unique_parameter_coordinates']==1728801536
                inputs[str(ap)],inputs[str(rp)]=sha(ap),sha(rp)
                current={s:a['losses'][s]['mean_NLL_FP64'] for s in ['BF16','same_fit_BF16','nominal_controls']}
                if controls is None: controls=current
                assert controls==current
                native.append(a['losses']['native']['mean_NLL_FP64'])
                row={k:a[k] for k in ['retained_scheduled_evaluations','warmup_scheduled_evaluations',
                    'sampled_coordinates_across_uses','max_temporary_weight_count_bytes']}
                row.update(seed=seed,logical_pbits_if_fully_resident=bits*1728801536,
                    samples_per_pbit=budget//bits,full_four_stage_case_wall_seconds=r['elapsed_seconds'])
                assert row['full_four_stage_case_wall_seconds']>0
                row['retained_evaluations_per_full_case_wall_second']=row['retained_scheduled_evaluations']/row['full_four_stage_case_wall_seconds']
                if bits!=1:
                    paired=cases['K1'][seed]
                    assert a['reference_single_audit_sha256']==paired['audit_sha256']
                    assert a['retained_scheduled_evaluations']==paired['retained']
                    assert a['warmup_scheduled_evaluations']==bits*paired['warmup']
                    assert a['word_bits']==bits and a['samples']==budget//bits
                records.append({'seed':seed,'audit_path':str(ap),'results_path':str(rp),'audit_sha256':sha(ap),
                    'retained':row['retained_scheduled_evaluations'],'warmup':row['warmup_scheduled_evaluations']})
                rows.append(row)
            key=f'K{bits}'
            deltas={s:summary([v-c for v in native]) for s,c in controls.items()}
            sf=deltas['same_fit_BF16']; passed=sf['MC95_t4_interval'][1]<=MARGIN
            methods[key]={'backend':backend,'native_NLL':summary(native),'controls_NLL':controls,
                'native_minus_own_control_NLL':deltas,'perplexity_exp_mean_NLL':math.exp(math.fsum(native)/5),
                'own_same_fit_BF16_perplexity':math.exp(controls['same_fit_BF16']),
                'PPL_ratio_to_own_same_fit_BF16':math.exp(sf['mean']),
                'PPL_ratio_MC95':list(map(math.exp,sf['MC95_t4_interval'])),
                'conditional_2percent_gate_passed':passed}
            costs[key]=rows;cases[key]={v['seed']:v for v in records}
            if passed and key not in minimum:
                minimum[key]={'retained_work_budget':budget,'samples_per_pbit':budget//bits,
                    'logical_pbits_per_parameter':bits,'backend':backend}
        contrasts={}
        for left,right in itertools.combinations(['K1','K4','K8'],2):
            contrasts[left+'_minus_'+right]={
                'raw_native_NLL':summary([a-b for a,b in zip(methods[left]['native_NLL']['values'],methods[right]['native_NLL']['values'])]),
                'own_BF16_gap_difference':summary([a-b for a,b in zip(methods[left]['native_minus_own_control_NLL']['same_fit_BF16']['values'],methods[right]['native_minus_own_control_NLL']['same_fit_BF16']['values'])]),
                'backends':[methods[left]['backend'],methods[right]['backend']]}
        cohorts.append({'budget':budget,'seeds':seeds,'methods':methods,'costs_per_seed':costs,'paired_root_key_contrasts':contrasts})
    protocol={'status':'post_execution_descriptive_aggregation_of_frozen_case_conditions',
        'source_sha256':sha(Path(__file__)),'independent_verifier_sha256':sha(Path(__file__).with_name('verify_broad_matched_statistics_v1.py')),
        'critical':CRITICAL,'quality_margin_NLL':MARGIN,'inputs_sha256':inputs,
        'scope':'No new sampling, model fitting, seed pooling or quality-margin changes. C1024 uses complete GB10 cohorts; higher-budget K1 is GB10 and K4/K8 are H100. Own-backend controls retained; cross-format contrasts are descriptive, not a claim of hardware-identical execution. All intervals conditional on the same fixed2048targets.'}
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    report={'status':'completed_matched_cases_aggregated_pending_independent_audit','protocol_sha256':sha(OUT/'protocol.json'),
        'cohorts':cohorts,'lowest_qualifying_budget_among_explored':minimum,
        'resource_scope':'Logical device counts assume complete residency. Scheduled node evaluations are not physical flips. Case wall time includes loading and all four inference stages, excludes separate auditing, and is simulator throughput, not device latency or energy.',
        'scope':protocol['scope']}
    (OUT/'statistics.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'lowest_qualifying':minimum}))


if __name__=='__main__':main()
