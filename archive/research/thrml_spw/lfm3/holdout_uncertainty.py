"""Paired article/MC uncertainty for the reserved, stratified text design.

Pure arithmetic only: no model, tokenizer, holdout files or scoring. Bootstrap
intervals are approximate diagnostics, not a guarantee of population coverage.
Five inference seeds are clusters shared across all articles in each replicate.
"""
import math
import numpy as np


T4_CRITICAL = 2.7764451051977987


def article_means(short_token_values, long_token_values):
    """Each of 40 articles contributes 256 targets; short uses four windows."""
    short = np.asarray(short_token_values, dtype=np.float64)
    long = np.asarray(long_token_values, dtype=np.float64)
    if short.shape != (5, 96, 64) or long.shape != (5, 16, 256):
        raise ValueError('Expected five MC clusters, 96x64 short and16x256 long targets')
    if not np.isfinite(short).all() or not np.isfinite(long).all():
        raise ValueError('All token values must be finite')
    return short.reshape(5, 24, 4, 64).mean(axis=(2, 3)), long.mean(axis=2)


def bootstrap_article_values(short, long, replicates, key):
    """Return short, long, and 60:40 pooled paired bootstrap gap estimates.

Resample five seed rows globally, and24/16 articles within their respective
strata. The SAME seed sample is used for both strata and all articles. Windows
within an article stay together. Length strata are different articles, so their
contrast must not be interpreted as a causal context-length effect.
"""
    short = np.asarray(short, dtype=np.float64)
    long = np.asarray(long, dtype=np.float64)
    if short.shape != (5, 24) or long.shape != (5, 16):
        raise ValueError('Expected seed by article matrices5x24 and5x16')
    if not np.isfinite(short).all() or not np.isfinite(long).all():
        raise ValueError('All article values must be finite')
    if not isinstance(replicates, int) or replicates < 2:
        raise ValueError('Require at least two bootstrap replicates')
    rng = np.random.default_rng(key)
    seeds = rng.integers(0, 5, size=(replicates, 5))
    short_articles = rng.integers(0, 24, size=(replicates, 24))
    long_articles = rng.integers(0, 16, size=(replicates, 16))
    short_draws = short[seeds[:, :, None], short_articles[:, None, :]].mean(axis=(1, 2))
    long_draws = long[seeds[:, :, None], long_articles[:, None, :]].mean(axis=(1, 2))
    return np.stack([short_draws, long_draws, .6 * short_draws + .4 * long_draws], axis=1)


def paired_summary(short_token_gaps, long_token_gaps, replicates=10000, key=202610400):
    """Summarize paired NLL gaps (candidate minus reference), never raw ratios.

Caller must first verify identical target IDs/model point and subtract paired
token losses. No pass/fail rule or treatment selection is performed here.
"""
    short, long = article_means(short_token_gaps, long_token_gaps)
    per_seed = np.stack([short.mean(1), long.mean(1), .6 * short.mean(1) + .4 * long.mean(1)], axis=1)
    draws = bootstrap_article_values(short, long, replicates, key)
    records = {}
    for i, name in enumerate(['short', 'long', 'pooled']):
        values = per_seed[:, i]
        center = float(values.mean())
        se = float(values.std(ddof=1) / math.sqrt(5))
        interval = np.quantile(draws[:, i], [.025, .975], method='linear').tolist()
        records[name] = {'per_seed_mean_NLL_gap': values.tolist(), 'mean_NLL_gap': center,
            'MC_standard_error_conditional_on_articles': se,
            'MC95_t4_interval_conditional_on_articles': [center - T4_CRITICAL * se, center + T4_CRITICAL * se],
            'stratified_crossed_bootstrap_percentile95_interval': interval,
            'PPL_ratio_exp_mean_gap': math.exp(center),
            'bootstrap_PPL_ratio_interval': [math.exp(v) for v in interval]}
    return {'contrasts': records, 'bootstrap_replicates': replicates, 'bootstrap_key': key,
        'articles_short_long': [24, 16], 'targets_per_article': 256, 'pooled_target_weights': [.6, .4],
        'scope': 'Pure paired uncertainty arithmetic for five MCclusters and40articleclusters withfixed60:40targetmix. Approximate product/crossed nonparametricbootstrap and conditional MCt4intervals; not formal coverage, training uncertainty, independence of all tokens, fullWikiTextperformance, pretraining-decontamination, causal context-length effect or BF16 equivalence.'}
