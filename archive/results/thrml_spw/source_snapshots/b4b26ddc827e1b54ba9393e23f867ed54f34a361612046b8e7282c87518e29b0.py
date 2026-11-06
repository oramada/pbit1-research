"""Five-seed paired quality accounting at one matched adapted-model budget."""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import numpy as np
from .adaptation_native_v2_statistics import summary


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def analyze(data, critical, margin):
    assert set(data) == {'common_controls', 'methods'}
    assert set(data['common_controls']) == {'BF16', 'same_fit_BF16'}
    assert set(data['methods']) == {'K1', 'K4', 'K8'}
    methods = {}
    for key in ['K1', 'K4', 'K8']:
        values = data['methods'][key]['native_NLL']
        assert len(values) == 5 and np.isfinite(values).all()
        nominal = data['methods'][key]['nominal_NLL']
        assert math.isfinite(nominal)
        native = summary(values, critical)
        controls = {**data['common_controls'], 'own_nominal': nominal}
        gaps = {name: summary(np.asarray(values) - value, critical) for name, value in controls.items()}
        methods[key] = {'native_NLL': native, 'native_minus_control_NLL': gaps,
                        'subset_perplexity': math.exp(native['mean']),
                        'over_same_fit_BF16_PPL_ratio': math.exp(gaps['same_fit_BF16']['mean']),
                        'conditional_BF16_gate_passed': gaps['same_fit_BF16']['MC95_t4_interval'][1] <= margin}
    contrasts = {}
    for left, right in itertools.combinations(['K1', 'K4', 'K8'], 2):
        contrasts[left + '_minus_' + right] = summary(
            np.asarray(data['methods'][left]['native_NLL']) - np.asarray(data['methods'][right]['native_NLL']), critical)
    return {'methods': methods, 'paired_NLL_contrasts': contrasts}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--budget', type=int, choices=[1024, 4096], required=True)
    args = parser.parse_args()
    root = Path('results/thrml_spw/lfm3_1p7b')
    path = root / 'adapted_word_validation_protocol.json'
    protocol = json.loads(path.read_text())
    for name, digest in protocol['sources_sha256'].items():
        assert sha(Path(__file__).with_name(name)) == digest
    inputs, methods, work = {}, {}, {}
    common = None
    single_audits = {}
    for bits in [1, 4, 8]:
        values, rows, nominal = [], [], None
        for seed in protocol['seeds_by_budget'][str(args.budget)]:
            if bits == 1:
                folder = root / ('counted_mean_validation' if args.budget == 1024 else 'counted_mean_t4096') / 'mean' / str(seed)
                status = 'all_direct_program_coverage_losses_and_resource_receipts_verified'
                verifier = protocol['reference_verifier_sha256'][str(args.budget)]
            else:
                folder = root / 'adapted_word_validation' / f'C{args.budget}' / f'K{bits}' / 'mean' / str(seed)
                status = 'all_adapted_word_coverage_losses_controls_and_matched_work_verified'
                verifier = protocol['sources_sha256']['verify_adapted_word_validation.py']
            audit_path, result_path = folder / 'independent_audit.json', folder / 'results.json'
            audit = json.loads(audit_path.read_text())
            assert audit['status'] == status and audit['source_sha256'] == verifier
            assert audit['seed'] == seed and audit['targets'] == 2048 and audit['unique_parameter_coordinates'] == 1728801536
            assert audit['results_sha256'] == sha(result_path)
            inputs[str(audit_path)], inputs[str(result_path)] = sha(audit_path), sha(result_path)
            current = {name: audit['losses'][name]['mean_NLL_FP64'] for name in ['BF16', 'same_fit_BF16']}
            if common is None:
                common = current
            assert current == common
            n = audit['losses']['nominal_controls']['mean_NLL_FP64']
            if nominal is None:
                nominal = n
            assert n == nominal
            values.append(audit['losses']['native']['mean_NLL_FP64'])
            rows.append({key: audit[key] for key in ['retained_scheduled_evaluations', 'warmup_scheduled_evaluations',
                         'sampled_coordinates_across_uses', 'max_temporary_weight_count_bytes']})
            if bits == 1:
                single_audits[seed] = (audit, sha(audit_path))
            else:
                paired, paired_hash = single_audits[seed]
                assert audit['reference_single_audit_sha256'] == paired_hash
                assert audit['budget'] == args.budget and audit['word_bits'] == bits and audit['samples'] == args.budget // bits
                assert audit['all_original_and_same_fit_BF16_controls_bitwise_equal']
                assert audit['retained_scheduled_evaluations'] == paired['retained_scheduled_evaluations']
                assert audit['warmup_scheduled_evaluations'] == bits * paired['warmup_scheduled_evaluations']
        methods[f'K{bits}'] = {'native_NLL': values, 'nominal_NLL': nominal}
        work[f'K{bits}'] = rows
    data = {'common_controls': common, 'methods': methods}
    result = {'status': 'all_five_paired_adapted_K1_K4_K8_cases_summarized', 'budget': args.budget,
              'protocol_sha256': sha(path), 'inputs_sha256': inputs, 'data': data, 'work_per_seed': work,
              **analyze(data, float(protocol['MC_t4_critical']), protocol['BF16_quality_gate_NLL_margin']),
              'scope': protocol['statistics_scope']}
    out = root / 'adapted_word_validation' / f'C{args.budget}' / 'statistics.json'
    with out.open('x') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')


if __name__ == '__main__':
    main()
