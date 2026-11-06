"""Article/seed bootstrap with explicit counts for the unexposed subset."""
import math
import numpy as np


def paired_summary(short_token_gaps, long_token_gaps, replicates=10000, key=202611699):
    short = np.asarray(short_token_gaps, dtype=np.float64)
    long = np.asarray(long_token_gaps, dtype=np.float64)
    if (short.ndim != 3 or long.ndim != 3 or short.shape[0] != 5 or long.shape[0] != 5
            or short.shape[2] != 64 or long.shape[2] != 256
            or short.shape[1] == 0 or short.shape[1] % 4 or long.shape[1] == 0):
        raise ValueError('Five seeds, four64-target windows per short article, one256-target long window required')
    if not np.isfinite(short).all() or not np.isfinite(long).all():
        raise ValueError('Finite paired token losses required')
    if type(replicates) is not int or replicates < 2:
        raise ValueError('At least two bootstrap replicates required')
    ns, nl = short.shape[1]//4, long.shape[1]
    s = short.reshape(5, ns, 4, 64).mean((2, 3))
    l = long.mean(2)
    ws, wl = ns/(ns+nl), nl/(ns+nl)
    rng = np.random.default_rng(key)
    seeds = rng.integers(0, 5, (replicates, 5))
    sa = rng.integers(0, ns, (replicates, ns))
    la = rng.integers(0, nl, (replicates, nl))
    sd = s[seeds[:, :, None], sa[:, None, :]].mean((1, 2))
    ld = l[seeds[:, :, None], la[:, None, :]].mean((1, 2))
    sv, lv = s.mean(1), l.mean(1)
    records = {}
    for name, values, draws in [('short', sv, sd), ('long', lv, ld),
            ('pooled', ws*sv+wl*lv, ws*sd+wl*ld)]:
        center = float(values.mean())
        se = float(values.std(ddof=1)/math.sqrt(5))
        interval = np.quantile(draws, [.025, .975], method='linear').tolist()
        records[name] = {'per_seed_mean_NLL_gap': values.tolist(), 'mean_NLL_gap': center,
            'MC_standard_error_conditional_on_articles': se,
            'MC95_t4_interval_conditional_on_articles': [center-2.7764451051977987*se, center+2.7764451051977987*se],
            'stratified_crossed_bootstrap_percentile95_interval': interval,
            'PPL_ratio_exp_mean_gap': math.exp(center),
            'bootstrap_PPL_ratio_interval': [math.exp(v) for v in interval]}
    return {'contrasts': records, 'bootstrap_replicates': replicates, 'bootstrap_key': key,
        'articles_short_long': [ns, nl], 'targets_per_article': 256,
        'pooled_target_weights': [ws, wl],
        'scope': 'Common globally resampled inference seeds crossed with separately resampled short/long articles. Approximate percentile bootstrap, not formal population or simultaneous coverage.'}


def unexposed_summary(loss_sets, parameters):
    excluded = parameters['excluded_short_article_indices']
    if excluded != [0, 1]:
        raise ValueError('Only the two documented previously exposed articles may be excluded')
    keep = [i for i in range(24) if i not in excluded]
    windows = [4*i+j for i in keep for j in range(4)]
    reports = {}
    for control in ['BF16', 'same_fit_BF16', 'nominal_controls']:
        reports[control] = paired_summary(
            np.asarray(loss_sets['native'][0])[:, windows]-np.asarray(loss_sets[control][0])[:, windows],
            np.asarray(loss_sets['native'][1])-np.asarray(loss_sets[control][1]),
            parameters['bootstrap_replicates'], parameters['unexposed_bootstrap_key'])
    checks = {}
    for name, record in reports['same_fit_BF16']['contrasts'].items():
        checks[name] = {
            'conditional_MC_upper_within_margin': record['MC95_t4_interval_conditional_on_articles'][1] <= parameters['NLL_margin'],
            'crossed_bootstrap_upper_within_margin': record['stratified_crossed_bootstrap_percentile95_interval'][1] <= parameters['NLL_margin']}
    return {'targets_per_seed': 9728, 'articles': 38, 'excluded_short_article_indices': excluded,
        'paired_contrasts': reports, 'primary_same_fit_BF16_tolerance_checks': checks,
        'quality_tolerance_met_in_both_strata_and_pooled': all(all(v.values()) for v in checks.values())}
