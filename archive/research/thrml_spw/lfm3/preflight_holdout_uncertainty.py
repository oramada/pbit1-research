"""Independent synthetic replay of article aggregation and bootstrap indices."""
import hashlib
import json
from pathlib import Path
import numpy as np
from .holdout_uncertainty import article_means, bootstrap_article_values, paired_summary


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    rng = np.random.default_rng(202610401)
    short_tokens = rng.normal(0, .1, (5, 96, 64))
    long_tokens = rng.normal(0, .2, (5, 16, 256))
    short, long = article_means(short_tokens, long_tokens)
    for seed in range(5):
        for article in range(24):
            expected = sum(float(x) for window in short_tokens[seed, 4 * article:4 * article + 4] for x in window) / 256
            np.testing.assert_allclose(short[seed, article], expected, rtol=0, atol=1e-15)
        for article in range(16):
            np.testing.assert_allclose(long[seed, article], sum(map(float, long_tokens[seed, article])) / 256, rtol=0, atol=1e-15)
    reps, key = 64, 202610402
    draws = bootstrap_article_values(short, long, reps, key)
    independent = np.random.default_rng(key)
    ss = independent.integers(0, 5, (reps, 5))
    aa = independent.integers(0, 24, (reps, 24))
    bb = independent.integers(0, 16, (reps, 16))
    for i in range(reps):
        a = sum(float(short[s, j]) for s in ss[i] for j in aa[i]) / 120
        b = sum(float(long[s, j]) for s in ss[i] for j in bb[i]) / 80
        np.testing.assert_allclose(draws[i], [a, b, .6 * a + .4 * b], rtol=0, atol=1e-15)
    report = paired_summary(short_tokens, long_tokens, reps, key)
    for i, name in enumerate(['short', 'long', 'pooled']):
        rec = report['contrasts'][name]
        expected = (short_tokens.mean() if i == 0 else long_tokens.mean() if i == 1 else
                    (short_tokens.sum() + long_tokens.sum()) / (5 * 10240))
        np.testing.assert_allclose(rec['mean_NLL_gap'], expected, rtol=0, atol=1e-15)
        np.testing.assert_array_equal(rec['stratified_crossed_bootstrap_percentile95_interval'], np.quantile(draws[:, i], [.025, .975], method='linear'))
    # Common MC noise must survive across all40articles and bothstrata;
    # independently resampling seeds per article would spuriously shrink it.
    shocks = np.arange(-2., 3.)
    common = bootstrap_article_values(np.repeat(shocks[:, None], 24, 1), np.repeat(shocks[:, None], 16, 1), 20000, key)
    np.testing.assert_allclose(common[:, 0], common[:, 1], rtol=0, atol=1e-15)
    np.testing.assert_allclose(common[:, 2], common[:, 0], rtol=0, atol=1e-15)
    expected_variance = float(shocks.var() / 5)
    assert abs(common[:, 2].var() - expected_variance) < .025
    # Zero seed variance does not erase variation betweenarticles.
    article_only = bootstrap_article_values(np.repeat(np.arange(24.)[None, :], 5, 0), np.repeat(np.arange(16.)[None, :], 5, 0), 20000, key)
    expected_article_var = .6 ** 2 * np.arange(24.).var() / 24 + .4 ** 2 * np.arange(16.).var() / 16
    assert abs(article_only[:, 2].var() - expected_article_var) < .06
    zero = paired_summary(np.zeros((5, 96, 64)), np.zeros((5, 16, 256)), reps, key)
    assert zero['contrasts']['pooled']['mean_NLL_gap'] == 0
    assert zero['contrasts']['pooled']['stratified_crossed_bootstrap_percentile95_interval'] == [0, 0]
    for a, b in [(short_tokens[:, :-1], long_tokens), (short_tokens * np.nan, long_tokens)]:
        try:
            article_means(a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid input accepted')
    root = Path('results/thrml_spw/lfm3_1p7b')
    source = Path(__file__)
    result = {'status': 'synthetic_article_grouping_exact_resample_replay_and_common_seed_noise_verified',
        'sources_sha256': {p.name: sha(p) for p in [source, source.with_name('holdout_uncertainty.py')]},
        'exact_replayed_replicates': reps, 'common_seed_variance_observed': float(common[:, 2].var()),
        'common_seed_variance_theory': expected_variance,
        'article_only_variance_observed': float(article_only[:, 2].var()),
        'article_only_variance_theory': expected_article_var,
        'scope': 'Synthetic arithmetic fixture only. No reserved token files, model predictions or task scores loaded. No treatment/statisticalgate selected and no formal interval coverage claimed.'}
    with (root / 'holdout_uncertainty_preflight.json').open('x') as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
