"""No-text fixtures for the complete and previously unexposed decision rules."""
from pathlib import Path
import copy
import hashlib
import json
import math
import numpy as np
from .holdout_native_statistics_v4 import summarize
from .verify_holdout_native_statistics_v4 import verify_summary
from .holdout_uncertainty_v2 import paired_summary
from .holdout_uncertainty import paired_summary as original_paired_summary


def main():
    parameters = {'NLL_margin': math.log(1.02), 'bootstrap_replicates': 10000,
        'bootstrap_key': 202610400, 'unexposed_bootstrap_key': 202611699,
        'excluded_short_article_indices': [0, 1], 'control_loss_absolute_tolerance': 1e-12}
    rng = np.random.default_rng(202611053)
    short = np.repeat(rng.uniform(1, 2, (1, 96, 64)), 5, axis=0)
    long = np.repeat(rng.uniform(1, 2, (1, 16, 256)), 5, axis=0)
    cases = []
    retained = None
    for name in ['zero', 'small_gap', 'exposed_only_degradation', 'long_degradation']:
        sg, lg = np.zeros_like(short), np.zeros_like(long)
        if name == 'small_gap':
            shocks = np.array([-.003, -.001, 0, .001, .003])[:, None, None]
            sg += .005 + shocks
            lg += .005 + shocks
        elif name == 'exposed_only_degradation':
            sg[:, :8] = 1
        elif name == 'long_degradation':
            lg[:] = .1
        sets = {'BF16': (short+.4, long+.4), 'same_fit_BF16': (short, long),
            'nominal_controls': (short-.1, long-.1), 'native': (short+sg, long+lg)}
        report = summarize(sets, parameters)
        assert verify_summary(report, sets, parameters)
        expected = name in ['zero', 'small_gap']
        assert report['quality_tolerance_met_for_full_and_previously_unexposed'] == expected
        if name == 'exposed_only_degradation':
            assert report['previously_unexposed']['quality_tolerance_met_in_both_strata_and_pooled']
            assert not report['quality_tolerance_met_in_both_strata_and_pooled']
        cases.append({'case': name, 'combined_gate': expected,
            'full_gate': report['quality_tolerance_met_in_both_strata_and_pooled'],
            'unexposed_gate': report['previously_unexposed']['quality_tolerance_met_in_both_strata_and_pooled']})
        if name == 'small_gap': retained = report, sets
    report, sets = retained
    rejected = []
    for field in ['targets', 'articles', 'weights', 'mean', 'interval', 'combined_gate']:
        forged = copy.deepcopy(report)
        fresh = forged['previously_unexposed']
        if field == 'targets': fresh['targets_per_seed'] = 10240
        elif field == 'articles': fresh['articles'] = 40
        elif field == 'weights': fresh['paired_contrasts']['same_fit_BF16']['pooled_target_weights'] = [.6, .4]
        elif field == 'mean': fresh['paired_contrasts']['same_fit_BF16']['contrasts']['pooled']['mean_NLL_gap'] += .01
        elif field == 'interval': fresh['paired_contrasts']['same_fit_BF16']['contrasts']['long']['stratified_crossed_bootstrap_percentile95_interval'][1] += .1
        else: forged['quality_tolerance_met_for_full_and_previously_unexposed'] = False
        try: verify_summary(forged, sets, parameters)
        except AssertionError: rejected.append(field)
        else: raise AssertionError('Forged summary accepted: '+field)
    # When given all24/16articles the generalized routine must reproduce the
    # already audited original arithmetic, including the exact random draw plan.
    gap_s = rng.normal(0, .01, short.shape)
    gap_l = rng.normal(0, .01, long.shape)
    a = paired_summary(gap_s, gap_l, 10000, 202610400)
    b = original_paired_summary(gap_s, gap_l, 10000, 202610400)
    for name in ['short', 'long', 'pooled']:
        for field, expected in b['contrasts'][name].items():
            np.testing.assert_allclose(a['contrasts'][name][field], expected, rtol=2e-12, atol=2e-12)
    # Metric-only machine reductions may differ by a handful of FP64 ulps.
    # Each seed's actual loss is still retained in the paired arithmetic.
    drifted = copy.deepcopy(sets)
    drifted['same_fit_BF16'][0][4, 0, 0] += 5e-13
    tolerant = summarize(drifted, parameters)
    assert verify_summary(tolerant, drifted, parameters)
    rejected_drift = []
    for stage in ['BF16', 'same_fit_BF16', 'nominal_controls']:
        bad = copy.deepcopy(sets)
        bad[stage][0][4, 0, 0] += 2e-12
        try: summarize(bad, parameters)
        except ValueError: pass
        else: raise AssertionError('Control drift above frozen tolerance accepted')
        try: verify_summary(report, bad, parameters)
        except AssertionError: rejected_drift.append(stage)
        else: raise AssertionError('Independent auditor accepted excess control drift')
    out = Path('results/thrml_spw/broad_verification_20261105/statistics_v4_preflight')
    out.mkdir(exist_ok=False)
    source = Path(__file__)
    names = [source.name, 'holdout_native_statistics_v4.py', 'verify_holdout_native_statistics_v4.py',
        'holdout_uncertainty_v2.py', 'holdout_uncertainty.py', 'holdout_native_statistics.py',
        'verify_holdout_native_statistics.py']
    result = {'status': 'full40_and_unexposed38_analysis_and_independent_auditor_fixtures_passed',
        'sources_sha256': {n: hashlib.sha256(source.with_name(n).read_bytes()).hexdigest() for n in names},
        'parameters': parameters, 'cases': cases, 'forged_reports_rejected': rejected,
        'original_full40_arithmetic_reproduced': True,
        'metric_reduction_drift_tolerated_without_replacing_seed_losses': 5e-13,
        'excess_control_drift_rejected_by_both_paths': rejected_drift,
        'scope': 'Synthetic arithmetic only, no model, heldout texts, or quality outcomes. Full40andunexposed38mustbothsatisfytheunchanged2percentcriterion.'}
    (out/'results.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
