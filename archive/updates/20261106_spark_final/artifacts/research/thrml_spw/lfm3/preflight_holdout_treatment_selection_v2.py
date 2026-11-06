"""Synthetic three-budget eligibility and forged-cohort rejection checks."""
import argparse
import copy
from decimal import Decimal
import hashlib
import json
from pathlib import Path
from .holdout_treatment_selection_v2 import select_treatment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sources = {}
    for name in ['holdout_treatment_selection_v2.py', 'preflight_holdout_treatment_selection_v2.py',
                 'holdout_treatment_selection.py', 'count_budget_continuation.py']:
        p = Path(__file__).with_name(name)
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        assert (Path('results/thrml_spw/source_snapshots') / (h + '.py')).read_bytes() == p.read_bytes()
        sources[name] = h
    policy = {'critical': '2.77644510519779866907365875095', 'margin': '0.01980262729617973',
              **{str(c): {'protocol_sha256': f'p{c}', 'verifier_sha256': f'v{c}', 'same_fit_BF16_NLL': 4.0}
                 for c in [1024, 4096, 8192]}}
    def fixture(c, passed):
        value = 4.0 if passed else 4.1
        difference = str(Decimal(str(value)) - Decimal(4))
        return {'statistics_sha256': f's{c}', 'audit': {
                    'status': 'all_five_mean_quality_statistics_and_workspace_resource_counts_verified',
                    'source_sha256': f'v{c}', 'statistics_sha256': f's{c}'},
                'statistics': {'status': 'all_five_prior_selected_mean_cases_summarized',
                    'protocol_sha256': f'p{c}', 'native_NLL': {'values': [value] * 5},
                    'controls_NLL': {'same_fit_BF16': 4.0}, 'conditional_same_fit_BF16_gate_passed': passed,
                    'native_minus_control_NLL': {'same_fit_BF16': {'MC95_t4_interval': [difference, difference]}}}}
    valid = 0
    for index, c in enumerate([1024, 4096, 8192]):
        pending = [fixture(b, False) if i < index else None for i, b in enumerate([1024, 4096, 8192])]
        assert select_treatment(*pending, policy)['status'] == f'pending_complete_T{c}_cohort'
        valid += 1
        pending[index] = fixture(c, True)
        chosen = select_treatment(*pending, policy)
        assert chosen['samples'] == c and chosen['pbits_per_parameter'] == 1 and not chosen['score_holdout']
        valid += 1
    assert 'none_of_three' in select_treatment(*(fixture(c, False) for c in [1024, 4096, 8192]), policy)['status']
    valid += 1
    rejected = 0
    for index, c in enumerate([1024, 4096, 8192]):
        for change in ['four_seeds', 'audit_status', 'audit_hash', 'verifier', 'protocol', 'gate', 'interval', 'control', 'nan']:
            bad = copy.deepcopy(fixture(c, True))
            if change == 'four_seeds': bad['statistics']['native_NLL']['values'].pop()
            elif change == 'audit_status': bad['audit']['status'] = 'pending'
            elif change == 'audit_hash': bad['audit']['statistics_sha256'] = 'changed'
            elif change == 'verifier': bad['audit']['source_sha256'] = 'changed'
            elif change == 'protocol': bad['statistics']['protocol_sha256'] = 'changed'
            elif change == 'gate': bad['statistics']['conditional_same_fit_BF16_gate_passed'] = False
            elif change == 'interval': bad['statistics']['native_minus_control_NLL']['same_fit_BF16']['MC95_t4_interval'][1] = .5
            elif change == 'control': bad['statistics']['controls_NLL']['same_fit_BF16'] = 3.0
            else: bad['statistics']['native_NLL']['values'][0] = 'NaN'
            bundles = [fixture(b, False) if i < index else None for i, b in enumerate([1024, 4096, 8192])]
            bundles[index] = bad
            try:
                select_treatment(*bundles, policy)
            except ValueError:
                rejected += 1
            else:
                raise AssertionError('Partial or forged cohort qualified')
    assert valid == 7 and rejected == 27
    with args.output.open('x') as handle:
        json.dump({'status': 'seven_three_budget_outcomes_and_27_invalid_cohorts_verified', 'sources_sha256': sources,
                   'valid_outcomes': valid, 'invalid_cohorts_rejected': rejected,
                   'scope': 'Synthetic guard only. No holdout tokens or neural outcomes read; full execution protocol still required.'}, handle, indent=2)
        handle.write('\n')


if __name__ == '__main__':
    main()
