"""Synthetic tests of accepted/pass/fail/missing/forged continuation inputs."""
import copy
import hashlib
import json
from pathlib import Path
from .count_budget_continuation import needs_higher_budget


def main():
    statistics = {'status': 'all_five_prior_selected_mean_cases_summarized', 'protocol_sha256': 'protocol',
        'native_NLL': {'values': [1, 2, 3, 4, 5]}, 'conditional_same_fit_BF16_gate_passed': True}
    audit = {'status': 'all_five_mean_quality_statistics_and_workspace_resource_counts_verified',
        'statistics_sha256': 'statistics'}
    assert needs_higher_budget(statistics, audit, 'statistics', 'protocol') is False
    failed = copy.deepcopy(statistics); failed['conditional_same_fit_BF16_gate_passed'] = False
    assert needs_higher_budget(failed, audit, 'statistics', 'protocol') is True
    invalid = []
    for key, value in [('status', 'running'), ('protocol_sha256', 'other'), ('conditional_same_fit_BF16_gate_passed', None),
                       ('native_NLL', {'values': [1]})]:
        item = copy.deepcopy(statistics); item[key] = value; invalid.append((item, audit))
    for key, value in [('status', 'pending'), ('statistics_sha256', 'forged')]:
        item = copy.deepcopy(audit); item[key] = value; invalid.append((statistics, item))
    for s, a in invalid:
        try: needs_higher_budget(s, a, 'statistics', 'protocol')
        except ValueError: pass
        else: raise AssertionError('Incomplete or unaudited cohort accepted')
    paths = [Path(__file__), Path(__file__).with_name('count_budget_continuation.py')]
    report = {'status': 'pass_fail_and_six_incomplete_or_forged_continuation_cases_verified',
        'sources_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        'scope': 'Synthetic predicate only; no native scores read and no quality conclusion.'}
    with Path('results/thrml_spw/lfm3_1p7b/count_budget_continuation_preflight.json').open('x') as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report))


if __name__ == '__main__':
    main()
