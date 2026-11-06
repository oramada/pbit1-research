"""Versioned pre-score eligibility, adding the prospectively frozen8192 condition.

No holdout tokens are opened and eligibility never itself permits inference.
"""
from .holdout_treatment_selection import cohort_passes


def select_treatment(lower, higher, highest, policy):
    candidates = [(1024, lower), (4096, higher), (8192, highest)]
    for samples, bundle in candidates:
        if bundle is None:
            return {'status': f'pending_complete_T{samples}_cohort', 'score_holdout': False}
        if cohort_passes(bundle, policy[str(samples)], policy['critical'], policy['margin']):
            return {'status': 'eligible_after_full_execution_protocol_freeze', 'score_holdout': False,
                    'pbits_per_parameter': 1, 'samples': samples, 'arm': 'mean'}
    return {'status': 'none_of_three_frozen_budgets_qualifies_continue_research_without_holdout_scoring', 'score_holdout': False}
