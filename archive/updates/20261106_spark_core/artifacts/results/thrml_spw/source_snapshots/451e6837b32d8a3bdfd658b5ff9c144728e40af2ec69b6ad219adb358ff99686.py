"""Synthetic complete-analysis checks; no reserved outcomes or token arrays."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from .holdout_native_statistics import summarize, STAGES


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    out = Path('results/thrml_spw/lfm3_1p7b/holdout_statistics_preflight')
    out.mkdir(exist_ok=False)
    parameters = {'NLL_margin': math.log(1.02), 'bootstrap_replicates': 10000, 'bootstrap_key': 202610400}
    records = []
    for name, short_gap, long_gap, shock, expected in [
        ('zero', 0., 0., 0., True), ('small_gap', .01, .01, 0., True),
        ('outside_margin', .03, .03, 0., False),
        ('pooled_masks_long_degradation', 0., .04, 0., False),
        ('uncertain_seeds', .01, .01, .05, False)]:
        control = (np.full((5,96,64), 4.), np.full((5,16,256), 4.))
        sets = {stage: control for stage in STAGES[:-1]}
        shifts = np.arange(-2., 3.)*shock
        sets['native'] = (control[0]+short_gap+shifts[:,None,None], control[1]+long_gap+shifts[:,None,None])
        r = summarize(sets, parameters)
        assert r['quality_tolerance_met_in_both_strata_and_pooled'] == expected
        paired = r['paired_contrasts']['same_fit_BF16']['contrasts']
        for label, gap in [('short',short_gap),('long',long_gap),('pooled',.6*short_gap+.4*long_gap)]:
            np.testing.assert_allclose(paired[label]['mean_NLL_gap'], gap, rtol=0, atol=2e-15)
        if name == 'pooled_masks_long_degradation':
            assert all(r['primary_same_fit_BF16_tolerance_checks']['pooled'].values())
            assert not all(r['primary_same_fit_BF16_tolerance_checks']['long'].values())
        raw = out/(name+'.npz')
        with raw.open('xb') as f:
            np.savez_compressed(f, native_short=sets['native'][0], native_long=sets['native'][1],
                control_short=control[0], control_long=control[1])
        records.append({'case': name, 'quality_tolerance_met': expected, 'synthetic_inputs_sha256': sha(raw),
            'short_gap': short_gap, 'long_gap': long_gap, 'MC_shocks': shifts.tolist()})
    rejected = []
    for name in ['missing_control', 'four_seeds', 'missing_long_targets', 'nonfinite', 'variable_control']:
        sets = {stage: (np.full((5,96,64), 4.), np.full((5,16,256), 4.)) for stage in STAGES}
        if name == 'missing_control': sets.pop('same_fit_BF16')
        elif name == 'four_seeds': sets['native'] = (sets['native'][0][:-1], sets['native'][1][:-1])
        elif name == 'missing_long_targets': sets['native'] = (sets['native'][0], sets['native'][1][:,:,:-1])
        elif name == 'nonfinite': sets['native'][0][0,0,0] = np.nan
        else: sets['BF16'][0][1,0,0] += 1
        try:
            summarize(sets, parameters)
        except ValueError:
            rejected.append(name)
        else:
            raise AssertionError('Invalid statistics inputs accepted')
    sources = {name: sha(Path(__file__).with_name(name)) for name in
        ['holdout_native_statistics.py','preflight_holdout_native_statistics.py','holdout_uncertainty.py','holdout_batch_layout.py']}
    report = {'status': 'five_synthetic_complete_analysis_outcomes_and_five_invalid_inputs_verified',
        'sources_sha256': sources, 'parameters': parameters, 'cases': records, 'invalid_inputs_rejected': rejected,
        'scope': 'Prospective arithmetic only. Tests zero/small/large gaps, long-stratum degradation masked by pooled mean, and shared MC shocks. No actual heldout data or predictions, formal confidence-coverage claim, frozen final analysis protocol, independent full statistics audit or new model inference.'}
    with (out/'results.json').open('x') as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report))


if __name__ == '__main__':
    main()
