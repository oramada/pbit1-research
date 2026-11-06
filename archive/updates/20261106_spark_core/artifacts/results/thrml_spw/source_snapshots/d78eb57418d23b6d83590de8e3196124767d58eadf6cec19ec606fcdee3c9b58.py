"""Decimal independent paired arithmetic and all native input hash checks."""
from decimal import Decimal, localcontext
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path('results/thrml_spw/lfm3_1p7b')
    out = root / 'adaptation_native_v2'
    p = json.loads((root / 'adaptation_native_v2_protocol.json').read_text())
    r = json.loads((out / 'statistics.json').read_text())
    assert r['status'] == 'all_three_native_validation_candidates_summarized'
    for name, digest in p['sources_sha256'].items():
        assert sha(Path(__file__).with_name(name)) == digest
    for path, digest in r['inputs_sha256'].items():
        assert sha(Path(path)) == digest
    def near(value, expected):
        actual = Decimal(str(value))
        assert abs(actual - expected) <= Decimal('2e-12') * max(Decimal(1), abs(expected))
    with localcontext() as ctx:
        ctx.prec = 50
        critical = Decimal(p['MC_t4_critical'])
        def check(rec, values):
            assert len(values) == 5
            for actual, expected in zip(rec['values'], values, strict=True):
                near(actual, expected)
            mean = sum(values) / 5
            variance = sum((v - mean) ** 2 for v in values) / 4
            se = (variance / 5).sqrt()
            near(rec['mean'], mean); near(rec['standard_error'], se)
            near(rec['MC95_t4_interval'][0], mean - critical * se)
            near(rec['MC95_t4_interval'][1], mean + critical * se)
            return mean, mean + critical * se
        data = {}
        for arm in p['arms']:
            values, expected, bf16, same_fit = [], [], [], []
            for seed in p['seeds']:
                a = json.loads((out / arm / str(seed) / 'independent_audit.json').read_text())
                values.append(Decimal(str(a['losses']['native']['mean_NLL_FP64'])))
                expected.append(Decimal(str(a['losses']['nominal_controls']['mean_NLL_FP64'])))
                bf16.append(Decimal(str(a['losses']['BF16']['mean_NLL_FP64'])))
                same_fit.append(Decimal(str(a['losses']['same_fit_BF16']['mean_NLL_FP64'])))
                assert a['retained_scheduled_evaluations'] == r['retained_scheduled_evaluations_per_seed']
                assert a['warmup_scheduled_evaluations'] == r['warmup_scheduled_evaluations_per_seed']
            assert len(set(expected)) == len(set(bf16)) == len(set(same_fit)) == 1
            rec = r['arms'][arm]
            near(rec['BF16_NLL'], bf16[0]); near(rec['learned_expected_operator_NLL'], expected[0])
            absolute, _ = check(rec['native_NLL'], values)
            gap, upper = check(rec['native_minus_BF16_NLL'], [v - bf16[0] for v in values])
            near(rec['same_fit_BF16_NLL'], same_fit[0])
            same_fit_gap, same_fit_upper = check(rec['native_minus_same_fit_BF16_NLL'], [v - same_fit[0] for v in values])
            check(rec['native_minus_expected_operator_NLL'], [v - expected[0] for v in values])
            near(rec['native_subset_perplexity_exp_mean_NLL'], absolute.exp())
            near(rec['native_over_BF16_PPL_ratio'], gap.exp())
            for actual, limit in zip(rec['native_over_BF16_PPL_ratio_MC95_interval'], rec['native_minus_BF16_NLL']['MC95_t4_interval'], strict=True):
                near(actual, Decimal(str(limit)).exp())
            near(rec['native_over_same_fit_BF16_PPL_ratio'], same_fit_gap.exp())
            for actual, limit in zip(rec['native_over_same_fit_BF16_PPL_ratio_MC95_interval'], rec['native_minus_same_fit_BF16_NLL']['MC95_t4_interval'], strict=True):
                near(actual, Decimal(str(limit)).exp())
            assert rec['conditional_BF16_quality_gate_passed'] == (same_fit_upper <= Decimal(str(p['BF16_quality_gate_NLL_margin'])))
            data[arm] = values
        for arm in ['mean', 'binomial']:
            check(r['paired_native_minus_unadapted_NLL'][arm], [a - b for a, b in zip(data[arm], data['baseline'], strict=True)])
    selection = json.loads((out / 'surrogate_selection.json').read_text())
    scores = {}
    for arm in p['arms']:
        path = root / 'adaptation_pilot' / arm / 'results.json'
        assert sha(path) == selection['inputs_sha256'][str(path)]
        train = json.loads(path.read_text())
        scores[arm] = sum(v['mean_NLL_FP64'] for v in train['after_binomial']) / 5
    chosen = min(p['arms'], key=lambda arm: scores[arm])
    assert chosen == selection['selected_arm'] == r['selected_by_prior_Torch_surrogate']
    assert r['selected_conditional_BF16_quality_gate_passed'] == r['arms'][chosen]['conditional_BF16_quality_gate_passed']
    audit = {'status': 'all_native_candidate_statistics_resource_counts_and_prior_selection_verified',
             'source_sha256': sha(Path(__file__)), 'statistics_sha256': sha(out / 'statistics.json'),
             'candidates': 3, 'MC_seeds_per_candidate': 5, 'targets_per_seed': 2048,
             'scope': 'Decimal paired statistical arithmetic, all upstream audit/selection hashes and source hashes checked. Underlying saved logits audited independently; no native full trajectory replay, data-generalization, equivalence or hardware proof.'}
    with (out / 'statistics_audit.json').open('x') as f:
        json.dump(audit, f, indent=2)
    print(json.dumps(audit))


if __name__ == '__main__':
    main()
