"""All-candidate paired native validation statistics, conditional on fixed data."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summary(values, critical):
    x = np.asarray(values, dtype=np.float64)
    mean = float(x.mean())
    se = float(x.std(ddof=1) / np.sqrt(len(x)))
    return {'values': x.tolist(), 'mean': mean, 'standard_error': se,
            'MC95_t4_interval': [mean - critical * se, mean + critical * se]}


def main():
    root = Path('results/thrml_spw/lfm3_1p7b')
    out = root / 'adaptation_native_v2'
    p = json.loads((root / 'adaptation_native_v2_protocol.json').read_text())
    inputs = {}
    def read(path):
        value = json.loads(path.read_text())
        inputs[str(path)] = sha(path)
        return value
    selection = read(out / 'surrogate_selection.json')
    critical = float(p['MC_t4_critical'])
    arms, work = {}, []
    for arm in p['arms']:
        nll, mean_control, bf16, same_fit = [], [], [], []
        for seed in p['seeds']:
            audit = read(out / arm / str(seed) / 'independent_audit.json')
            assert audit['status'] == 'all_direct_program_coverage_losses_and_resource_receipts_verified'
            assert audit['targets'] == 2048 and audit['arm'] == arm and audit['seed'] == seed
            nll.append(audit['losses']['native']['mean_NLL_FP64'])
            mean_control.append(audit['losses']['nominal_controls']['mean_NLL_FP64'])
            bf16.append(audit['losses']['BF16']['mean_NLL_FP64'])
            same_fit.append(audit['losses']['same_fit_BF16']['mean_NLL_FP64'])
            work.append((audit['retained_scheduled_evaluations'], audit['warmup_scheduled_evaluations']))
        assert len(set(mean_control)) == len(set(bf16)) == len(set(same_fit)) == 1
        gap = summary(np.asarray(nll) - bf16[0], critical)
        noise = summary(np.asarray(nll) - mean_control[0], critical)
        same_fit_gap = summary(np.asarray(nll) - same_fit[0], critical)
        absolute = summary(nll, critical)
        arms[arm] = {'native_NLL': absolute, 'BF16_NLL': bf16[0], 'learned_expected_operator_NLL': mean_control[0],
                     'same_fit_BF16_NLL': same_fit[0], 'native_minus_same_fit_BF16_NLL': same_fit_gap,
                     'native_minus_BF16_NLL': gap, 'native_minus_expected_operator_NLL': noise,
                     'native_subset_perplexity_exp_mean_NLL': math.exp(absolute['mean']),
                     'native_over_BF16_PPL_ratio': math.exp(gap['mean']),
                     'native_over_BF16_PPL_ratio_MC95_interval': [math.exp(v) for v in gap['MC95_t4_interval']],
                     'native_over_same_fit_BF16_PPL_ratio': math.exp(same_fit_gap['mean']),
                     'native_over_same_fit_BF16_PPL_ratio_MC95_interval': [math.exp(v) for v in same_fit_gap['MC95_t4_interval']],
                     'conditional_BF16_quality_gate_passed': same_fit_gap['MC95_t4_interval'][1] <= p['BF16_quality_gate_NLL_margin']}
    assert len(set(work)) == 1
    assert len({v['BF16_NLL'] for v in arms.values()}) == 1
    baseline = np.asarray(arms['baseline']['native_NLL']['values'])
    contrasts = {arm: summary(np.asarray(arms[arm]['native_NLL']['values']) - baseline, critical) for arm in ['mean', 'binomial']}
    selected = selection['selected_arm']
    report = {'status': 'all_three_native_validation_candidates_summarized', 'arms': arms,
              'paired_native_minus_unadapted_NLL': contrasts, 'selected_by_prior_Torch_surrogate': selected,
              'selected_conditional_BF16_quality_gate_passed': arms[selected]['conditional_BF16_quality_gate_passed'],
              'retained_scheduled_evaluations_per_seed': work[0][0], 'warmup_scheduled_evaluations_per_seed': work[0][1],
              'MC_t4_critical': p['MC_t4_critical'], 'inputs_sha256': inputs,
              'scope': 'Five MC seed clusters on the same32validationcontexts/2048targets. All candidates reported, with selection fixed before native evaluation by earlier Torch surrogate scores on the same validation data. Quality gate uses exact same adapted expected weights rounded to BF16 with conventional operators and no ACT12, isolating representation loss from training gains. Original unadapted BF16 comparisons retained separately. Approximate t4 MC intervals condition on this data; no data-generalization coverage, confirmatory multiplicity control or BF16 numerical equivalence. The2percentPPL gate is exploratory and requires fresh-data assessment if passed. Scheduled node evaluations are not measured flips, energy or hardware latency.'}
    with (out / 'statistics.json').open('x') as f:
        json.dump(report, f, indent=2)
    print(json.dumps({arm: v['native_NLL']['mean'] for arm, v in arms.items()}))


if __name__ == '__main__':
    main()
