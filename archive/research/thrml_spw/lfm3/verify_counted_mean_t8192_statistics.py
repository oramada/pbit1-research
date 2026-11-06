"""Independent Decimal arithmetic for selected-program quality and resources."""
from decimal import Decimal, localcontext
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path('results/thrml_spw/lfm3_1p7b'); out = root / 'counted_mean_t8192'
    p = json.loads((root / 'counted_mean_t8192_protocol.json').read_text()); r = json.loads((out / 'statistics.json').read_text())
    assert r['status'] == 'all_five_prior_selected_mean_cases_summarized'
    assert r['protocol_sha256'] == sha(root / 'counted_mean_t8192_protocol.json')
    for name, digest in p['sources_sha256'].items(): assert sha(Path(__file__).with_name(name)) == digest
    for name, digest in r['inputs_sha256'].items(): assert sha(Path(name)) == digest
    selection = json.loads((root / 'adaptation_native_v2/surrogate_selection.json').read_text())
    assert selection['selected_arm'] == 'mean' and sha(root / 'adaptation_native_v2/surrogate_selection.json') == p['selection_sha256']
    values = []
    for i, seed in enumerate(p['seeds']):
        a = json.loads((out / 'mean' / str(seed) / 'independent_audit.json').read_text())
        assert a['status'] == 'all_direct_program_coverage_losses_and_resource_receipts_verified'
        assert a['targets'] == 2048 and a['unique_parameter_coordinates'] == 1728801536
        values.append(Decimal(str(a['losses']['native']['mean_NLL_FP64'])))
        assert r['work_per_seed'][i] == {k: a[k] for k in r['work_per_seed'][i]}
        for control, value in r['controls_NLL'].items(): assert value == a['losses'][control]['mean_NLL_FP64']
    def near(actual, expected):
        assert abs(Decimal(str(actual)) - expected) <= Decimal('2e-12') * max(Decimal(1), abs(expected))
    with localcontext() as ctx:
        ctx.prec = 50; critical = Decimal(p['MC_t4_critical'])
        def check(rec, observations):
            assert len(observations) == len(rec['values']) == 5
            for actual, expected in zip(rec['values'], observations, strict=True): near(actual, expected)
            mean = sum(observations) / 5
            se = (sum((v - mean) ** 2 for v in observations) / 20).sqrt()
            near(rec['mean'], mean); near(rec['standard_error'], se)
            near(rec['MC95_t4_interval'][0], mean - critical * se)
            near(rec['MC95_t4_interval'][1], mean + critical * se)
            return mean, mean + critical * se
        mean, _ = check(r['native_NLL'], values); near(r['native_subset_perplexity'], mean.exp())
        for control, baseline in r['controls_NLL'].items():
            gap, upper = check(r['native_minus_control_NLL'][control], [v - Decimal(str(baseline)) for v in values])
            if control == 'same_fit_BF16':
                near(r['native_over_same_fit_BF16_PPL_ratio'], gap.exp())
                for actual, limit in zip(r['native_over_same_fit_BF16_PPL_ratio_MC95_interval'], r['native_minus_control_NLL'][control]['MC95_t4_interval'], strict=True): near(actual, Decimal(str(limit)).exp())
                assert r['conditional_same_fit_BF16_gate_passed'] == (upper <= Decimal(str(p['BF16_quality_gate_NLL_margin'])))
    audit = {'status': 'all_five_mean_quality_statistics_and_workspace_resource_counts_verified',
        'source_sha256': sha(Path(__file__)), 'statistics_sha256': sha(out / 'statistics.json'), 'scope': p['scope']}
    with (out / 'statistics_audit.json').open('x') as f: json.dump(audit, f, indent=2)
    print(json.dumps({'status': audit['status']}))


if __name__ == '__main__':
    main()
