"""Prospective complete five-seed, paired article/MC holdout analysis.

No execution permission: main reads only completed independently audited cases.
Statistical protocol must be separately frozen before any reserved scoring.
"""
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from .holdout_batch_layout import assemble_losses
from .holdout_uncertainty import paired_summary


STAGES = ['BF16', 'same_fit_BF16', 'nominal_controls', 'native']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize(loss_sets, parameters):
    if set(loss_sets) != set(STAGES):
        raise ValueError('All four complete inference stages required')
    margin = float(parameters['NLL_margin'])
    if not math.isfinite(margin) or margin < 0:
        raise ValueError('Finite nonnegative margin required')
    arrays, controls, absolute = {}, {}, {}
    for stage in STAGES:
        short, long = (np.asarray(a, dtype=np.float64) for a in loss_sets[stage])
        if short.shape != (5, 96, 64) or long.shape != (5, 16, 256):
            raise ValueError('All five seeds and all reserved targets required')
        if not np.isfinite(short).all() or not np.isfinite(long).all():
            raise ValueError('Finite losses required')
        if stage != 'native' and (not np.all(short == short[0]) or not np.all(long == long[0])):
            raise ValueError('Deterministic control losses changed across seeds')
        arrays[stage] = short, long
        per_seed = (short.sum((1, 2))+long.sum((1, 2)))/10240
        mean = float(per_seed.mean())
        absolute[stage] = {'per_seed_NLL': per_seed.tolist(), 'mean_NLL': mean,
            'PPL_exp_mean_NLL': math.exp(mean), 'short_mean_NLL': float(short.mean()),
            'long_mean_NLL': float(long.mean())}
    for stage in STAGES[:-1]:
        controls[stage] = paired_summary(arrays['native'][0]-arrays[stage][0],
            arrays['native'][1]-arrays[stage][1], replicates=parameters['bootstrap_replicates'],
            key=parameters['bootstrap_key'])
    checks = {}
    primary = controls['same_fit_BF16']['contrasts']
    for name in ['short', 'long', 'pooled']:
        c = primary[name]
        checks[name] = {'conditional_MC_upper_within_margin': c['MC95_t4_interval_conditional_on_articles'][1] <= margin,
            'crossed_bootstrap_upper_within_margin': c['stratified_crossed_bootstrap_percentile95_interval'][1] <= margin}
    return {'absolute_NLL': absolute, 'paired_contrasts': controls,
        'primary_same_fit_BF16_tolerance_checks': checks,
        'quality_tolerance_met_in_both_strata_and_pooled': all(all(v.values()) for v in checks.values()),
        'NLL_margin': margin,
        'scope': 'Five fixed MC seeds crossed with24short/16long article clusters,256targets perarticle,60:40target mix. Paired NLL differences; diagnostic approximate percentile bootstrap and conditional MCt4 intervals. Conservative planned tolerance requires both upper bounds within margin for short,long andpooled. No formal simultaneous coverage, causal context-length effect, numerical BF16 equality, pretraining decontamination or physical-performance claim.'}


def main():
    root = Path('results/thrml_spw/lfm3_1p7b')
    protocol_path = root/'holdout_native_protocol.json'
    analysis_path = root/'holdout_native_statistics_protocol.json'
    p, analysis = json.loads(protocol_path.read_text()), json.loads(analysis_path.read_text())
    assert p['status'] == 'complete_execution_and_statistics_protocol_frozen_before_scoring'
    assert analysis['status'] == 'complete_paired_holdout_analysis_frozen_before_scoring'
    assert sha(analysis_path) == p['statistics_protocol_sha256']
    for name, digest in analysis['sources_sha256'].items():
        assert sha(Path(__file__).with_name(name)) == digest
    assert len(p['seeds']) == len(set(p['seeds'])) == 5
    assert p['targets'] == 10240 and p['articles'] == 40
    sets = {stage: [[], []] for stage in STAGES}
    bindings, resources, cases = {}, [], []
    for seed in p['seeds']:
        directory = root/'holdout_native'/str(seed)
        rp, ap = directory/'results.json', directory/'independent_audit.json'
        r, a = json.loads(rp.read_text()), json.loads(ap.read_text())
        assert a['status'] == 'full_reserved_case_raw_losses_controls_coverage_and_resources_independently_verified'
        assert a['source_sha256'] == p['independent_case_verifier_sha256'] and a['results_sha256'] == sha(rp)
        assert a['shared_protocol_sha256'] == sha(protocol_path)
        assert a['seed'] == r['seed'] == seed and a['samples'] == r['samples'] == p['samples']
        assert a['pbits_per_parameter'] == r['pbits_per_parameter'] == 1
        assert a['targets'] == r['targets'] == 10240 and a['articles'] == r['articles'] == 40
        for stage in STAGES:
            records = [{'batch_index': x['batch_index'], 'token_NLL': x['token_NLL']}
                for x in r['metrics'] if x['stage'] == stage]
            short, long = assemble_losses(p['batch_plan'], records)
            mean = (short.sum()+long.sum())/10240
            np.testing.assert_allclose(mean, a['losses'][stage]['mean_NLL_FP64'], rtol=1e-12, atol=1e-12)
            sets[stage][0].append(short); sets[stage][1].append(long)
        bindings[str(rp)] = sha(rp); bindings[str(ap)] = sha(ap)
        resources.append({key: a[key] for key in ['retained_scheduled_evaluations', 'warmup_scheduled_evaluations',
            'sampled_coordinates_across_uses', 'max_temporary_weight_count_bytes', 'max_simultaneous_sampler_state_slots']})
        cases.append({'seed': seed, 'results_sha256': sha(rp), 'audit_sha256': sha(ap)})
    assert all(r == resources[0] for r in resources)
    summary = summarize({stage: tuple(np.stack(v) for v in arrays) for stage, arrays in sets.items()}, analysis['parameters'])
    report = {'status': 'complete_five_seed_reserved_statistics_produced_pending_independent_audit',
        'execution_protocol_sha256': sha(protocol_path), 'analysis_protocol_sha256': sha(analysis_path),
        'sources_sha256': analysis['sources_sha256'], 'inputs_sha256': bindings,
        'seeds': p['seeds'], 'samples': p['samples'], 'pbits_per_parameter': 1,
        'unique_parameter_coordinates': 1728801536, 'targets_per_seed': 10240, 'articles': 40,
        'cases': cases, 'resources_per_seed': resources, **summary}
    with (root/'holdout_native/statistics.json').open('x') as f:
        json.dump(report, f, indent=2)
    print(json.dumps({'status': report['status'], 'quality_tolerance_met_in_both_strata_and_pooled':
        report['quality_tolerance_met_in_both_strata_and_pooled']}))


if __name__ == '__main__':
    main()
