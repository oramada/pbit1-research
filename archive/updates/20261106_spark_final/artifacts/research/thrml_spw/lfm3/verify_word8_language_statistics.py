"""Decimal audit of all three frozen representations, contrasts and resource costs."""
from decimal import Decimal, localcontext
import hashlib
import itertools
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path('results/thrml_spw/lfm3_1p7b')
    out = root / 'word8_language'
    p = json.loads((root / 'word8_language_protocol.json').read_text())
    r = json.loads((out / 'statistics.json').read_text())
    assert r['status'] == 'all_five_K8_seeds_and_fifteen_three_method_contrasts_reported'
    assert r['source_sha256'] == sha(Path(__file__).with_name('word8_language_statistics.py'))
    for name, digest in p['sources_sha256'].items():
        assert sha(Path(__file__).with_name(name)) == digest
    for path, digest in r['inputs_sha256'].items():
        assert sha(Path(path)) == digest
    def near(actual, expected):
        x = Decimal(str(actual))
        assert abs(x - expected) <= Decimal('2e-12') * max(Decimal(1), abs(expected))
    with localcontext() as context:
        context.prec = 50
        def check_summary(rec, values):
            assert len(values) == len(rec['per_seed_NLL']) == 5
            mean = sum(values) / 5
            se = (sum((v - mean) ** 2 for v in values) / 20).sqrt()
            half = Decimal('2.7764451051977987') * se
            for actual, expected in zip(rec['per_seed_NLL'], values, strict=True):
                near(actual, expected)
            near(rec['mean_NLL'], mean); near(rec['MC_standard_error'], se)
            near(rec['MC95_t4_interval'][0], mean - half); near(rec['MC95_t4_interval'][1], mean + half)
            near(rec['exp_mean_NLL'], mean.exp())
            near(rec['MC95_exp_interval'][0], (mean - half).exp())
            near(rec['MC95_exp_interval'][1], (mean + half).exp())
        expected_methods = {'single_T1024': ('language_native_v2', 1, 1024),
            'word4_T256': ('word_language', 4, 256), 'word8_T128': ('word8_language', 8, 128)}
        assert set(r['methods']) == set(r['resources']) == set(expected_methods)
        values, resource_records = {}, []
        for name, (namespace, bits, samples) in expected_methods.items():
            data, expected = [], []
            retained = warmup = 0
            for seed in p['seeds']:
                folder = root / namespace / str(seed)
                result = json.loads((folder / 'results.json').read_text(), parse_float=Decimal)
                audit = json.loads((folder / 'independent_audit.json').read_text())
                assert audit['status'] == 'all_unique_parameters_and_saved_metrics_verified'
                assert audit['results_sha256'] == sha(folder / 'results.json')
                assert audit['seed'] == seed and audit['samples'] == samples and audit['unique_parameters_converted'] == 1728801536
                native = next(v for v in result['metrics'] if v['stage'] == 'native')
                nominal = next(v for v in result['metrics'] if v['stage'] == 'nominal_controls')
                losses = [v for context in native['token_NLL'] for v in context]
                assert len(losses) == 512
                data.append(sum(losses) / 512)
                expected.append(sum(v for context in nominal['token_NLL'] for v in context) / 512)
                retained += audit['retained_evaluations_including_padding']; warmup += audit['warmup_evaluations_including_padding']
                resource_records.append((name, seed, audit['retained_evaluations_including_padding'], audit['warmup_evaluations_including_padding']))
            values[name] = data
            check_summary(r['methods'][name], data)
            assert len(set(expected)) == 1
            near(r['methods'][name]['nominal_expected_operator_NLL'], expected[0])
            resource = r['resources'][name]
            assert resource['pbits_per_parameter'] == bits and resource['samples'] == samples
            assert bits * samples == 1024
            assert resource['retained_evaluations_cohort'] == retained and resource['warmup_evaluations_cohort'] == warmup
            assert resource['logical_pbits_if_fully_resident'] == bits * 1728801536
            assert resource['programming_levels_per_node'] == (4096 if bits == 1 else 2)
            command_bits = 12 if bits == 1 else bits
            assert resource['conditional_packed_state_command_and_FP32ranges_bytes'] * 8 == 1728801536 * (bits + command_bits) + 64 * 13531430
        assert len(r['seed_records']) == 15
        for rec, (method, seed, retained, warmup) in zip(r['seed_records'], resource_records, strict=True):
            assert rec['method'] == method and rec['seed'] == seed and rec['retained'] == retained and rec['warmup'] == warmup
            near(rec['native_NLL'], values[method][p['seeds'].index(seed)])
        for seed in p['seeds']:
            rows = [v for v in r['seed_records'] if v['seed'] == seed]
            assert len({v['retained'] for v in rows}) == 1
            assert rows[1]['warmup'] == 4 * rows[0]['warmup'] and rows[2]['warmup'] == 8 * rows[0]['warmup']
        pairs = list(itertools.combinations(expected_methods, 2))
        assert set(r['pairwise_NLL_differences']) == {a + '_minus_' + b for a, b in pairs}
        for first, second in pairs:
            check_summary(r['pairwise_NLL_differences'][first + '_minus_' + second],
                          [a - b for a, b in zip(values[first], values[second], strict=True)])
        control = json.loads((root / 'language_controls_batch/results.json').read_text(), parse_float=Decimal)
        expected_names = set()
        for name in expected_methods:
            for rec in control['metrics']:
                key = name + '_minus_' + rec['condition']
                expected_names.add(key)
                baseline = sum(v for context in rec['token_NLL'] for v in context) / 512
                check_summary(r['deterministic_control_contrasts'][key], [v - baseline for v in values[name]])
        assert set(r['deterministic_control_contrasts']) == expected_names and len(expected_names) == 12
    audit = {'status': 'all_three_methods_fifteen_contrasts_and_matched_resources_verified',
        'source_sha256': sha(Path(__file__)), 'statistics_sha256': sha(out / 'statistics.json'),
        'MC_seeds_per_method': 5, 'fixed_targets_per_seed': 512,
        'scope': 'Independent Decimal arithmetic from audited token losses, all15contrasts, full-coordinate coverage receipts, resource1:4:8warmup andequalretainedwork. Conditional MC only; no full-model rawtrajectoryreplay, data-generalization, equivalence or hardware proof.'}
    with (out / 'statistics_audit.json').open('x') as f:
        json.dump(audit, f, indent=2)
    print(json.dumps(audit))


if __name__ == '__main__':
    main()
