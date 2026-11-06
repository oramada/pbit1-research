"""Independently replay bounded allocation and actual-input cost inequalities."""
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path('results/thrml_spw/lfm3_1p7b')
    out = root / 'layer_allocation_candidates'
    p = json.loads((out / 'protocol.json').read_text())
    r = json.loads((out / 'results.json').read_text())
    assert sha(Path(__file__).with_name('extend_layer_allocation.py')) == p['source_sha256']
    assert sha(root / 'layer_calibration/results.json') == p['calibration_results_sha256']
    assert sha(root / 'layer_calibration/independent_audit.json') == p['calibration_audit_sha256']
    assert sha(root / 'adaptation_pilot/mean/program_package.json') == r['upstream_program_package_sha256']
    c = json.loads((root / 'layer_calibration/results.json').read_text())
    scores, geometry = c['scores'], c['mean_costs']
    lookup = 'model.embed_tokens'
    costs = {**geometry, lookup: 1028 * 2048}
    cap = 1024 * sum(costs.values())
    policy = {name: 1024 if name == lookup else 16 for name in scores}
    spent = sum(costs[name] * policy[name] for name in scores)
    while True:
        possible = []
        for name in sorted(scores):
            old = policy[name]
            if old == 8192 or scores[name] <= 0:
                continue
            additional = costs[name] * old
            if spent + additional <= cap:
                decrease = scores[name] / old - scores[name] / (old * 2)
                possible.append((-decrease / additional, name, additional))
        if not possible:
            break
        _, name, amount = min(possible)
        policy[name] *= 2; spent += amount
    e = r['extended']
    assert e['samples'] == policy and e['conservative_costs'] == costs
    assert e['conservative_cost'] == spent <= e['conservative_reference_cap'] == cap
    assert e['conservative_unused_cap'] == cap - spent
    assert e['actual_calibration_cost'] == sum(geometry[name] * policy[name] for name in scores)
    assert policy[lookup] >= 1024 and max(policy.values()) <= 8192
    assert 2047 * 128 * max(policy.values()) < 2 ** 31
    expected = {'uniform': {name: 1024 for name in scores},
                'fixed_lookup_max4096': c['proposal']['greedy_samples'], 'bounded_lookup_max8192': policy}
    assert set(r['candidates']) == set(expected)
    values = {}
    for label, counts in expected.items():
        assert counts == r['candidates'][label]['samples']
        values[label] = sum(scores[name] / counts[name] for name in sorted(scores))
        assert abs(values[label] - r['candidates'][label]['proxy']) <= 1e-12 * max(values[label], 1e-30)
    chosen = min(values, key=lambda name: values[name])
    assert r['selected_policy'] == chosen and r['selected_samples'] == expected[chosen]
    assert abs(r['selected_proxy_ratio_to_uniform'] - values[chosen] / values['uniform']) < 1e-12
    # Every possible distinct-token count, both selected and rawextendedpolicy.
    for counts in [policy, expected[chosen]]:
        assert counts[lookup] >= 1024
        for unique in range(1029):
            nodes = {**geometry, lookup: unique * 2048}
            actual = sum(nodes[name] * counts[name] for name in scores)
            reference = sum(nodes.values()) * 1024
            assert actual <= reference
    assert r['supported_batch_token_limit'] == e['maximum_lookup_tokens'] == 1028
    audit = {'status': 'bounded_lookup_greedy_policy_proxy_and_all_1029_input_cost_bounds_verified',
        'source_sha256': sha(Path(__file__)), 'results_sha256': sha(out / 'results.json'),
        'selected_policy': chosen, 'selected_proxy_ratio_to_uniform': r['selected_proxy_ratio_to_uniform'],
        'maximum_samples': max(expected[chosen].values()), 'lookup_samples': expected[chosen][lookup],
        'scope': 'Independent greedyobjective/budget replay and actual-nodework inequality at every0..1028distincttoken count; nominalproxy algebra audited upstream. Not globaloptimality or sampledquality/latency/energy/BF16proof. T8192nativepublicpreflight still required.'}
    with (out / 'independent_audit.json').open('x') as f:
        json.dump(audit, f, indent=2)
    print(json.dumps(audit))


if __name__ == '__main__':
    main()
