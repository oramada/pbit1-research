"""Synthetic holdout eligibility guards; no model result or holdout token access."""
import argparse
import copy
from decimal import Decimal
import hashlib
import json
from pathlib import Path
from .holdout_treatment_selection import select_treatment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sources = {}
    for name in ['holdout_treatment_selection.py', 'preflight_holdout_treatment_selection.py', 'count_budget_continuation.py']:
        p = Path(__file__).with_name(name)
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        assert (Path('results/thrml_spw/source_snapshots') / (digest + '.py')).read_bytes() == p.read_bytes()
        sources[name] = digest
    policy = {'critical': '2.77644510519779866907365875095', 'margin': '0.01980262729617973',
              '1024': {'protocol_sha256': 'low-protocol', 'verifier_sha256': 'low-verifier', 'same_fit_BF16_NLL': 4.0},
              '4096': {'protocol_sha256': 'high-protocol', 'verifier_sha256': 'high-verifier', 'same_fit_BF16_NLL': 4.0}}
    def fixture(budget, passed):
        value = 4.0 if passed else 4.1
        difference = str(Decimal(str(value)) - Decimal(4))
        return {'statistics_sha256': f'stats-{budget}',
                'statistics': {'status': 'all_five_prior_selected_mean_cases_summarized',
                    'protocol_sha256': policy[str(budget)]['protocol_sha256'],
                    'native_NLL': {'values': [value] * 5}, 'controls_NLL': {'same_fit_BF16': 4.0},
                    'native_minus_control_NLL': {'same_fit_BF16': {'MC95_t4_interval': [difference, difference]}},
                    'conditional_same_fit_BF16_gate_passed': passed},
                'audit': {'status': 'all_five_mean_quality_statistics_and_workspace_resource_counts_verified',
                    'source_sha256': policy[str(budget)]['verifier_sha256'], 'statistics_sha256': f'stats-{budget}'}}
    assert select_treatment(None, None, policy)['status'] == 'pending_complete_lower_cohort'
    assert select_treatment(fixture(1024, True), None, policy)['samples'] == 1024
    assert select_treatment(fixture(1024, False), None, policy)['status'] == 'pending_complete_higher_cohort'
    assert select_treatment(fixture(1024, False), fixture(4096, True), policy)['samples'] == 4096
    assert 'neither_frozen_budget' in select_treatment(fixture(1024, False), fixture(4096, False), policy)['status']
    rejected = 0
    for change in ['four_seeds', 'audit_status', 'audit_hash', 'verifier', 'protocol', 'gate', 'interval', 'control', 'nan']:
        bad = copy.deepcopy(fixture(1024, True))
        if change == 'four_seeds': bad['statistics']['native_NLL']['values'].pop()
        elif change == 'audit_status': bad['audit']['status'] = 'pending'
        elif change == 'audit_hash': bad['audit']['statistics_sha256'] = 'changed'
        elif change == 'verifier': bad['audit']['source_sha256'] = 'changed'
        elif change == 'protocol': bad['statistics']['protocol_sha256'] = 'changed'
        elif change == 'gate': bad['statistics']['conditional_same_fit_BF16_gate_passed'] = False
        elif change == 'interval': bad['statistics']['native_minus_control_NLL']['same_fit_BF16']['MC95_t4_interval'][1] = .5
        elif change == 'control': bad['statistics']['controls_NLL']['same_fit_BF16'] = 3.0
        else: bad['statistics']['native_NLL']['values'][0] = 'NaN'
        try:
            select_treatment(bad, None, policy)
        except ValueError:
            rejected += 1
        else:
            raise AssertionError('Ineligible/forged cohort qualified')
    assert rejected == 9
    with args.output.open('x') as handle:
        json.dump({'status': 'five_eligibility_outcomes_and_nine_partial_or_forged_cohorts_verified',
                   'sources_sha256': sources, 'valid_outcomes': 5, 'invalid_cohorts_rejected': rejected,
                   'scope': 'Synthetic guard only. Eligibility never itself permits scoring: full holdout execution/statistics protocol still required. No holdout tokens or model scores read.'}, handle, indent=2)
        handle.write('\n')


if __name__ == '__main__':
    main()
