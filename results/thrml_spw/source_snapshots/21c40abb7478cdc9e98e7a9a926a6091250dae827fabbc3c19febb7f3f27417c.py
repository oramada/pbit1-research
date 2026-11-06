"""Pre-score treatment eligibility for the frozen, untouched article holdout.

This pure function neither opens holdout tokens nor executes inference. A full
execution/statistics protocol must still be frozen before any holdout scoring.
"""
from decimal import Decimal, localcontext
from .count_budget_continuation import needs_higher_budget


def cohort_passes(bundle, expected, critical, margin):
    statistics, audit = bundle['statistics'], bundle['audit']
    failed = needs_higher_budget(statistics, audit, bundle['statistics_sha256'], expected['protocol_sha256'])
    if audit.get('source_sha256') != expected['verifier_sha256']:
        raise ValueError('Unexpected independent statistics verifier')
    with localcontext() as ctx:
        ctx.prec = 70
        values = [Decimal(str(v)) for v in statistics['native_NLL']['values']]
        baseline = Decimal(str(statistics['controls_NLL']['same_fit_BF16']))
        t, tolerance = Decimal(str(critical)), Decimal(str(margin))
        if not all(v.is_finite() for v in [*values, baseline, t, tolerance]):
            raise ValueError('Finite statistics required')
        if baseline != Decimal(str(expected['same_fit_BF16_NLL'])):
            raise ValueError('Unexpected adapted-point BF16 control')
        mean = sum(values) / Decimal(5)
        se = (sum((v - mean) ** 2 for v in values) / Decimal(4) / Decimal(5)).sqrt()
        exact_upper = mean - baseline + t * se
        recorded = statistics['native_minus_control_NLL']['same_fit_BF16']['MC95_t4_interval']
        if len(recorded) != 2 or not all(Decimal(str(v)).is_finite() for v in recorded):
            raise ValueError('Complete finite interval required')
        exact_lower = mean - baseline - t * se
        if any(abs(Decimal(str(v)) - expected_value) > Decimal('1e-12') for v, expected_value in zip(recorded, [exact_lower, exact_upper])):
            raise ValueError('Recorded interval disagrees with all five seeds')
        passed = exact_upper <= tolerance
        if passed != (not failed):
            raise ValueError('Gate outcome disagrees with independently recomputed upper bound')
        return passed


def select_treatment(lower_bundle, higher_bundle, policy):
    if lower_bundle is None:
        return {'status': 'pending_complete_lower_cohort', 'score_holdout': False}
    if cohort_passes(lower_bundle, policy['1024'], policy['critical'], policy['margin']):
        return {'status': 'eligible_after_full_execution_protocol_freeze', 'score_holdout': False,
                'pbits_per_parameter': 1, 'samples': 1024, 'arm': 'mean'}
    if higher_bundle is None:
        return {'status': 'pending_complete_higher_cohort', 'score_holdout': False}
    if cohort_passes(higher_bundle, policy['4096'], policy['critical'], policy['margin']):
        return {'status': 'eligible_after_full_execution_protocol_freeze', 'score_holdout': False,
                'pbits_per_parameter': 1, 'samples': 4096, 'arm': 'mean'}
    return {'status': 'neither_frozen_budget_qualifies_continue_research_without_holdout_scoring', 'score_holdout': False}
