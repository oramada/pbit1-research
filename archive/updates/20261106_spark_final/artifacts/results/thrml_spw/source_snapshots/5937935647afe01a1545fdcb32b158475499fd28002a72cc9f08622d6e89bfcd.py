"""Independent Decimal arithmetic, exact controls and batch resource checks."""
from decimal import Decimal, localcontext
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda: f.read(16 * 1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def main():
    root = Path('results/thrml_spw/lfm3_1p7b')
    out = root / 'allocated_native'
    p = json.loads((root / 'allocated_native_protocol.json').read_text())
    r = json.loads((out / 'statistics.json').read_text())
    assert r['status'] == 'all_five_paired_allocated_uniform_cases_summarized'
    assert r['protocol_sha256'] == sha(root / 'allocated_native_protocol.json')
    for name, digest in p['sources_sha256'].items():
        assert sha(Path(__file__).with_name(name)) == digest
    for name, digest in r['inputs_sha256'].items():
        assert sha(Path(name)) == digest
    def near(actual, expected):
        assert abs(Decimal(str(actual)) - expected) <= Decimal('2e-12') * max(Decimal(1), abs(expected))
    data = {'allocated': [], 'uniform': []}
    checks = []
    for i, seed in enumerate(p['seeds']):
        cases = {}
        for method, folder in [('allocated', 'allocated_native'), ('uniform', 'adaptation_native_v2')]:
            case = root / folder / 'mean' / str(seed)
            audit = json.loads((case / 'independent_audit.json').read_text())
            result = json.loads((case / 'results.json').read_text())
            assert audit['results_sha256'] == sha(case / 'results.json')
            assert audit['status'] == 'all_direct_program_coverage_losses_and_resource_receipts_verified'
            assert result['package_sha256'] == p['mean_program_package_sha256']
            for stage in ['BF16', 'same_fit_BF16', 'nominal_controls']:
                assert audit['losses'][stage]['mean_NLL_FP64'] == r['controls_NLL'][stage]
            data[method].append(Decimal(str(audit['losses']['native']['mean_NLL_FP64'])))
            assert r['work_per_seed'][method][i] == {k: audit[k] for k in r['work_per_seed'][method][i]}
            cases[method] = (audit, result)
        a, u = cases['allocated'][1], cases['uniform'][1]
        assert a['same_fit_BF16_weight_hashes'] == u['same_fit_BF16_weight_hashes']
        for batch, (oa, ou) in enumerate(zip(a['outputs'], u['outputs'], strict=True)):
            assert sha(Path(oa['path'])) == oa['sha256'] and sha(Path(ou['path'])) == ou['sha256']
            with np.load(oa['path']) as aa, np.load(ou['path']) as uu:
                for key in ['input_ids', 'BF16', 'same_fit_BF16', 'nominal_controls']:
                    np.testing.assert_array_equal(aa[key], uu[key])
            ca = [v for v in a['operator_calls'] if v['batch'] == batch and v['stage'] == 'native']
            cu = [v for v in u['operator_calls'] if v['batch'] == batch and v['stage'] == 'native']
            assert len(ca) == len(cu) == 287
            for x, y in zip(ca, cu, strict=True):
                for key in ['module', 'canonical_program', 'program_sha256', 'call', 'weight_coordinates', 'padding_nodes', 'warmup_evaluations', 'embedding_rows']:
                    assert x[key] == y[key]
            aw, uw = (sum(v['retained_evaluations'] for v in cc) for cc in [ca, cu])
            assert aw <= uw
            checks.append({'seed': seed, 'batch': batch, 'allocated_retained': aw, 'uniform_retained': uw})
    with localcontext() as ctx:
        ctx.prec = 50
        critical = Decimal(p['MC_t4_critical'])
        def check(rec, values):
            assert len(values) == len(rec['values']) == 5
            for actual, expected in zip(rec['values'], values, strict=True):
                near(actual, expected)
            mean = sum(values) / 5
            se = (sum((v - mean) ** 2 for v in values) / 20).sqrt()
            near(rec['mean'], mean); near(rec['standard_error'], se)
            near(rec['MC95_t4_interval'][0], mean - critical * se)
            near(rec['MC95_t4_interval'][1], mean + critical * se)
            return mean, mean + critical * se
        for method, values in data.items():
            rec = r['arms'][method]
            absolute, _ = check(rec['native_NLL'], values)
            gap, upper = check(rec['native_minus_same_fit_BF16_NLL'], [v - Decimal(str(r['controls_NLL']['same_fit_BF16'])) for v in values])
            near(rec['native_subset_perplexity'], absolute.exp())
            near(rec['native_over_same_fit_BF16_PPL_ratio'], gap.exp())
            assert rec['conditional_BF16_quality_gate_passed'] == (upper <= Decimal(str(p['BF16_quality_gate_NLL_margin'])))
        check(r['paired_allocated_minus_uniform_NLL'], [a - u for a, u in zip(data['allocated'], data['uniform'], strict=True)])
    allocation = json.loads((root / 'layer_allocation_candidates/results.json').read_text())['selected_samples']
    assert r['serial_operator_sample_cycles'] == p['serial_operator_sample_cycles'] == {'allocated': sum(allocation.values()), 'uniform': 287 * 1024}
    audit = {'status': 'all_paired_statistics_exact_controls_and_actual_batch_resource_bounds_verified',
        'source_sha256': sha(Path(__file__)), 'statistics_sha256': sha(out / 'statistics.json'),
        'batch_checks': checks, 'scope': p['scope']}
    with (out / 'statistics_audit.json').open('x') as f:
        json.dump(audit, f, indent=2)
    print(json.dumps({'status': audit['status']}))


if __name__ == '__main__':
    main()
