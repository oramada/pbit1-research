"""All-five-seed paired analysis on the frozen exploratory language subset."""
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path('results/thrml_spw/lfm3_1p7b')
    out = root / 'language_native_v2'
    protocol = json.loads((root / 'language_native_v2_protocol.json').read_text())
    pipeline = json.loads((out / 'pipeline.json').read_text())
    assert pipeline['status'] == 'all_five_seeds_independently_verified'
    assert pipeline['completed_seeds'] == protocol['seeds']
    controls = json.loads((root / 'language_controls_batch/results.json').read_text())
    control_audit = root / 'language_controls_batch/independent_audit.json'
    assert sha(control_audit) == protocol['controls_audit_sha256']
    inputs = {str(control_audit): sha(control_audit),
              str(root / 'language_controls_batch/results.json'): sha(root / 'language_controls_batch/results.json'),
              str(root / 'language_native_v2_protocol.json'): sha(root / 'language_native_v2_protocol.json'),
              str(out / 'pipeline.json'): sha(out / 'pipeline.json')}
    native = []
    nominal = []
    seed_rows = []
    retained = []
    warmup = []
    for seed in protocol['seeds']:
        run = out / str(seed)
        audit = json.loads((run / 'independent_audit.json').read_text())
        assert audit['status'] == 'all_unique_parameters_and_saved_metrics_verified'
        assert audit['seed'] == seed and audit['unique_parameters_converted'] == 1728801536
        assert sha(run / 'results.json') == audit['results_sha256']
        assert audit['matched_BF16_controls_exact']
        if seed != protocol['seeds'][0]:
            assert audit['matched_first_seed_deterministic_controls']
        r = json.loads((run / 'results.json').read_text())
        metrics = {m['stage']: m for m in r['metrics']}
        assert set(metrics) == {'identity_bridge', 'nominal_controls', 'native'}
        token_losses = np.asarray(metrics['native']['token_NLL'], dtype=np.float64)
        assert token_losses.shape == (8, 64) and np.isfinite(token_losses).all()
        native.append(token_losses)
        nominal.append(np.asarray(metrics['nominal_controls']['token_NLL']))
        seed_rows.append({'seed': seed, 'NLL': float(token_losses.mean()),
                          'subset_perplexity': float(np.exp(token_losses.mean())),
                          'context_mean_NLL': token_losses.mean(1).tolist()})
        retained.append(audit['retained_evaluations_including_padding'])
        warmup.append(audit['warmup_evaluations_including_padding'])
        for name in ['independent_audit.json', 'results.json', 'protocol.json']:
            inputs[str(run / name)] = sha(run / name)
    for a in nominal[1:]:
        np.testing.assert_array_equal(a, nominal[0])
    assert len(set(retained)) == len(set(warmup)) == 1
    values = np.asarray(native)
    seed_means = values.mean((1, 2))
    assert len(seed_means) == 5
    critical = 2.7764451051977987  # Student-t two-sided95%, df4.
    se = float(seed_means.std(ddof=1) / np.sqrt(5))
    center = float(seed_means.mean())
    contrasts = []
    for control in controls['metrics']:
        base = np.asarray(control['token_NLL'], dtype=np.float64)
        assert base.shape == (8, 64)
        paired = (values - base[None, :, :]).mean((1, 2))
        delta = float(paired.mean())
        standard_error = float(paired.std(ddof=1) / np.sqrt(5))
        interval = [delta - critical * standard_error, delta + critical * standard_error]
        contrasts.append({'control': control['condition'], 'control_NLL': float(base.mean()),
                          'per_seed_paired_NLL_difference': paired.tolist(),
                          'mean_paired_NLL_difference': delta,
                          'MC_standard_error': standard_error,
                          'MC95_t4_NLL_difference_interval': interval,
                          'perplexity_ratio_exp_mean_NLL_difference': float(np.exp(delta)),
                          'MC95_perplexity_ratio_interval': np.exp(interval).tolist()})
    summary = {'status': 'all_five_seeds_all_four_controls_reported',
               'source_sha256': sha(Path(__file__)), 'inputs_sha256': inputs,
               'seeds': seed_rows, 'seed_count': 5, 'predicted_tokens_per_seed': 512,
               'native_mean_NLL': center, 'native_MC_standard_error': se,
               'native_MC95_t4_NLL_interval': [center - critical * se, center + critical * se],
               'native_subset_perplexity_exp_mean_NLL': float(np.exp(center)),
               'nominal_control_NLL': float(nominal[0].mean()), 'contrasts': contrasts,
               'per_seed_retained_node_evaluations': retained[0],
               'per_seed_warmup_node_evaluations': warmup[0],
               'cohort_retained_node_evaluations': sum(retained),
               'cohort_warmup_node_evaluations': sum(warmup),
               'scope': 'MC interval conditional on these fixed512 tokens, this checkpoint, programming policy and sample budget. Shared per-operator draws across batch/context positions; five seeds are independent MC clusters. Not a confidence interval for unseen-text generalization, standard WikiText perplexity, equivalence proof or measured hardware benefit. All four deterministic baselines retained; dequantized BF16 controls are quality references, not matched stochastic resources.'}
    with (out / 'statistics.json').open('x') as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: summary[k] for k in ['status', 'native_mean_NLL', 'native_subset_perplexity_exp_mean_NLL', 'contrasts']}))


if __name__ == '__main__':
    main()
