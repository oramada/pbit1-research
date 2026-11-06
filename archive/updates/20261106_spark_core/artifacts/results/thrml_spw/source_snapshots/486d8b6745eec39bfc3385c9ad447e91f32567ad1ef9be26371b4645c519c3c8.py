"""Decimal paired-statistics oracle and independent artifact/resource binding."""
import argparse
from decimal import Decimal, localcontext
import hashlib
import itertools
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_math(result, data, critical, margin):
    with localcontext() as ctx:
        ctx.prec = 70
        t, m = Decimal(str(critical)), Decimal(str(margin))
        def vector(values):
            assert len(values) == 5
            return [Decimal(str(v)) for v in values]
        def check(observed, values):
            mean = sum(values) / Decimal(5)
            se = (sum((v - mean) ** 2 for v in values) / Decimal(4) / Decimal(5)).sqrt()
            interval = [mean - t * se, mean + t * se]
            assert len(observed['values']) == 5
            for actual, expected in zip(observed['values'], values):
                assert abs(Decimal(str(actual)) - expected) < Decimal('1e-12')
            for actual, expected in [(observed['mean'], mean), (observed['standard_error'], se), *zip(observed['MC95_t4_interval'], interval)]:
                assert abs(Decimal(str(actual)) - expected) < Decimal('1e-12')
            assert len(observed['MC95_t4_interval']) == 2
            return mean, interval
        assert set(result['methods']) == {'K1', 'K4', 'K8'}
        for key in ['K1', 'K4', 'K8']:
            values = vector(data['methods'][key]['native_NLL'])
            observed = result['methods'][key]
            mean, _ = check(observed['native_NLL'], values)
            assert abs(Decimal(str(observed['subset_perplexity'])) - mean.exp()) < Decimal('1e-9')
            controls = {**data['common_controls'], 'own_nominal': data['methods'][key]['nominal_NLL']}
            assert set(observed['native_minus_control_NLL']) == set(controls)
            for name, control in controls.items():
                gap, interval = check(observed['native_minus_control_NLL'][name], [v - Decimal(str(control)) for v in values])
                if name == 'same_fit_BF16':
                    assert observed['conditional_BF16_gate_passed'] == (interval[1] <= m)
                    assert abs(Decimal(str(observed['over_same_fit_BF16_PPL_ratio'])) - gap.exp()) < Decimal('1e-12')
        names = {a + '_minus_' + b for a, b in itertools.combinations(['K1', 'K4', 'K8'], 2)}
        assert set(result['paired_NLL_contrasts']) == names
        for left, right in itertools.combinations(['K1', 'K4', 'K8'], 2):
            lv, rv = vector(data['methods'][left]['native_NLL']), vector(data['methods'][right]['native_NLL'])
            check(result['paired_NLL_contrasts'][left + '_minus_' + right], [a - b for a, b in zip(lv, rv)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--budget', type=int, choices=[1024, 4096], required=True)
    args = parser.parse_args()
    root = Path('results/thrml_spw/lfm3_1p7b')
    protocol_path = root / 'adapted_word_validation_protocol.json'
    protocol = json.loads(protocol_path.read_text())
    for name, digest in protocol['sources_sha256'].items():
        assert sha(Path(__file__).with_name(name)) == digest
    directory = root / 'adapted_word_validation' / f'C{args.budget}'
    path = directory / 'statistics.json'
    result = json.loads(path.read_text())
    assert result['protocol_sha256'] == sha(protocol_path) and result['budget'] == args.budget
    assert result['status'] == 'all_five_paired_adapted_K1_K4_K8_cases_summarized'
    expected_inputs = {}
    reconstructed = {'common_controls': None, 'methods': {}}
    work = {}
    singles = {}
    for bits in [1, 4, 8]:
        native, costs, nominal = [], [], None
        for seed in protocol['seeds_by_budget'][str(args.budget)]:
            folder = (root / ('counted_mean_validation' if args.budget == 1024 else 'counted_mean_t4096') if bits == 1 else directory / f'K{bits}') / 'mean' / str(seed)
            audit_path, result_path = folder / 'independent_audit.json', folder / 'results.json'
            audit = json.loads(audit_path.read_text())
            assert audit['seed'] == seed and audit['targets'] == 2048 and audit['unique_parameter_coordinates'] == 1728801536
            assert audit['results_sha256'] == sha(result_path)
            if bits == 1:
                assert audit['status'] == 'all_direct_program_coverage_losses_and_resource_receipts_verified'
                assert audit['source_sha256'] == protocol['reference_verifier_sha256'][str(args.budget)]
                singles[seed] = audit, sha(audit_path)
            else:
                assert audit['status'] == 'all_adapted_word_coverage_losses_controls_and_matched_work_verified'
                assert audit['source_sha256'] == protocol['sources_sha256']['verify_adapted_word_validation.py']
                paired, paired_sha = singles[seed]
                assert audit['word_bits'] == bits and audit['budget'] == args.budget and audit['samples'] * bits == args.budget
                assert audit['reference_single_audit_sha256'] == paired_sha
                assert audit['all_original_and_same_fit_BF16_controls_bitwise_equal'] and audit['retained_work_exactly_matches_same_budget_single']
                assert audit['retained_scheduled_evaluations'] == paired['retained_scheduled_evaluations']
                assert audit['warmup_scheduled_evaluations'] == bits * paired['warmup_scheduled_evaluations']
                assert audit['sampled_coordinates_across_uses'] == paired['sampled_coordinates_across_uses']
            expected_inputs[str(audit_path)], expected_inputs[str(result_path)] = sha(audit_path), sha(result_path)
            common = {key: audit['losses'][key]['mean_NLL_FP64'] for key in ['BF16', 'same_fit_BF16']}
            if reconstructed['common_controls'] is None:
                reconstructed['common_controls'] = common
            assert common == reconstructed['common_controls']
            value = audit['losses']['nominal_controls']['mean_NLL_FP64']
            if nominal is None:
                nominal = value
            assert nominal == value
            native.append(audit['losses']['native']['mean_NLL_FP64'])
            costs.append({key: audit[key] for key in ['retained_scheduled_evaluations', 'warmup_scheduled_evaluations',
                          'sampled_coordinates_across_uses', 'max_temporary_weight_count_bytes']})
        reconstructed['methods'][f'K{bits}'] = {'native_NLL': native, 'nominal_NLL': nominal}
        work[f'K{bits}'] = costs
    assert expected_inputs == result['inputs_sha256'] and reconstructed == result['data'] and work == result['work_per_seed']
    verify_math(result, reconstructed, protocol['MC_t4_critical'], protocol['BF16_quality_gate_NLL_margin'])
    with (directory / 'statistics_audit.json').open('x') as handle:
        json.dump({'status': 'all_paired_adapted_spatial_statistics_inputs_Decimal_math_and_work_verified',
                   'source_sha256': sha(Path(__file__)), 'statistics_sha256': sha(path), 'budget': args.budget,
                   'scope': protocol['statistics_scope']}, handle, indent=2)
        handle.write('\n')


if __name__ == '__main__':
    main()
