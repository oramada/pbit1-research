"""Pure continuation predicate: never mistakes an incomplete cohort for failure."""


def needs_higher_budget(statistics, audit, statistics_sha256, expected_protocol_sha256):
    if audit.get('status') != 'all_five_mean_quality_statistics_and_workspace_resource_counts_verified':
        raise ValueError('Complete independently audited lower-budget statistics required')
    if audit.get('statistics_sha256') != statistics_sha256:
        raise ValueError('Lower statistics hash is not audited')
    if statistics.get('status') != 'all_five_prior_selected_mean_cases_summarized':
        raise ValueError('Incomplete lower-budget cohort')
    if statistics.get('protocol_sha256') != expected_protocol_sha256:
        raise ValueError('Unexpected lower-budget protocol')
    if len(statistics.get('native_NLL', {}).get('values', [])) != 5:
        raise ValueError('All five lower-budget seeds required')
    passed = statistics.get('conditional_same_fit_BF16_gate_passed')
    if type(passed) is not bool:
        raise ValueError('Explicit accepted gate outcome required')
    return not passed
