"""Extend the audited proxy candidate with a conservative variable-lookup bound."""
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bounded_proposal(scores, original_costs, lookup, maximum_tokens=1028, hidden=2048):
    names = sorted(scores)
    costs = dict(original_costs)
    costs[lookup] = maximum_tokens * hidden
    assert costs[lookup] >= original_costs[lookup]
    reference = 1024
    counts = {name: reference if name == lookup else 16 for name in names}
    cap = reference * sum(costs.values())
    used = sum(costs[name] * counts[name] for name in names)
    while True:
        candidates = []
        for index, name in enumerate(names):
            old = counts[name]
            if old >= 8192 or scores[name] <= 0:
                continue
            extra = costs[name] * old
            if used + extra <= cap:
                candidates.append((scores[name] / (2 * old) / extra, -index, name, extra))
        if not candidates:
            break
        _, _, name, extra = max(candidates)
        counts[name] *= 2; used += extra
    assert counts[lookup] >= reference
    assert 2047 * 128 * max(counts.values()) < 2 ** 31
    return {'samples': counts, 'conservative_costs': costs, 'conservative_cost': used,
            'conservative_reference_cap': cap, 'conservative_unused_cap': cap - used,
            'proxy': sum(scores[name] / counts[name] for name in names),
            'maximum_lookup_tokens': maximum_tokens,
            'actual_calibration_cost': sum(original_costs[name] * counts[name] for name in names)}


def main():
    root = Path('results/thrml_spw/lfm3_1p7b')
    out = root / 'layer_allocation_candidates'
    out.mkdir(exist_ok=True)
    audit = json.loads((root / 'layer_calibration/independent_audit.json').read_text())
    assert audit['status'] == 'all_module_use_variance_coefficients_raw_gradients_and_budget_candidate_verified'
    calibrated = json.loads((root / 'layer_calibration/results.json').read_text())
    assert audit['results_sha256'] == sha(root / 'layer_calibration/results.json')
    p = {'source_sha256': sha(Path(__file__)),
        'calibration_results_sha256': sha(root / 'layer_calibration/results.json'),
        'calibration_audit_sha256': sha(root / 'layer_calibration/independent_audit.json'),
        'method': 'Additional greedycandidate over16..8192samples, lookupminimum1024, othersminimum16. Replace lookupcost with conservativeM=1028tokens*2048weights/token. Referencecap1024*(sumothercosts+M). Select lowest calibrationproxy amonguniform,previousfixedlookupcandidate,andextendedcandidate; all retained. No native/validation/test scores used.',
        'known_calibration_observation': 'Lookup accounts for27.25percent offirst-ordercoefficient and was fixed1024bytheoriginalcandidate; lifting thatrestrictionmayreduceproxy. No sampledquality gain presumed.',
        'proof': 'Let m<=M beactual lookupnodes and TE>=1024. BoundC=sum_other(n*T)+M*TE <=1024*(sum_other(n)+M). Then actual_alloc-actual_uniform <=(m-M)*(TE-1024)<=0. This is an upperbudget inequality, not exact equality. Applies to batches withB*L<=1028tokens, including8x65and4x257; allothernodecostsunchanged.',
        'integer_guard': 'MaxsignedACT12*maxgroupsize*8192=2047*128*8192 <2**31; directpublic/nativepreflight at8192 still required beforeinference.',
        'scope': 'Postprocessing of audited gradientproxy calibrated onunusedtrainingwindows. Not optimality, true nonlinear lossvariance, expected taskquality, native gain or BF16 equivalence. Existinguniform/K8/adaptationstudies unchanged.'}
    with (out / 'protocol.json').open('x') as f:
        json.dump(p, f, indent=2)
    scores, costs = calibrated['scores'], calibrated['mean_costs']
    lookup = 'model.embed_tokens'
    extended = bounded_proposal(scores, costs, lookup)
    fixed = calibrated['proposal']
    candidates = {'uniform': {'samples': {name: 1024 for name in scores}, 'proxy': fixed['uniform_proxy']},
        'fixed_lookup_max4096': {'samples': fixed['greedy_samples'], 'proxy': fixed['greedy_proxy']},
        'bounded_lookup_max8192': {'samples': extended['samples'], 'proxy': extended['proxy']}}
    chosen = min(candidates, key=lambda name: candidates[name]['proxy'])
    selected = candidates[chosen]
    # Check every possible distinct-token count under the declared bound,
    # not just the two observed calibration batches.
    for tokens in range(1029):
        actual_costs = {**costs, lookup: tokens * 2048}
        allocated = sum(actual_costs[name] * selected['samples'][name] for name in scores)
        reference = 1024 * sum(actual_costs.values())
        assert allocated <= reference
    result = {'status': 'three_proxy_candidates_retained_and_shape_budget_bound_checked',
        'candidates': candidates, 'selected_policy': chosen, 'selected_samples': selected['samples'],
        'selected_proxy_ratio_to_uniform': selected['proxy'] / fixed['uniform_proxy'],
        'extended': extended, 'supported_batch_token_limit': 1028,
        'upstream_program_package_sha256': sha(root / 'adaptation_pilot/mean/program_package.json'),
        'scope': p['scope']}
    with (out / 'results.json').open('x') as f:
        json.dump(result, f, indent=2)
    print(json.dumps({'status': result['status'], 'selected': chosen,
        'proxy_ratio': result['selected_proxy_ratio_to_uniform'], 'lookup_samples': selected['samples'][lookup]}))


if __name__ == '__main__':
    main()
