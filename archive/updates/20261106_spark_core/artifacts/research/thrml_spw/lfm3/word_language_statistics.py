"""Paired five-seed single/word quality and separate resource dimensions."""
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def contrast(name, differences):
    values = np.asarray(differences, dtype=np.float64)
    assert values.shape == (5,)
    center = float(values.mean())
    se = float(values.std(ddof=1) / np.sqrt(5))
    half = 2.7764451051977987 * se
    interval = [center - half, center + half]
    return {'name': name, 'per_seed_NLL_difference': values.tolist(),
            'mean_NLL_difference': center, 'MC_standard_error': se,
            'MC95_t4_NLL_difference_interval': interval,
            'perplexity_ratio_exp_mean_difference': float(np.exp(center)),
            'MC95_perplexity_ratio_interval': np.exp(interval).tolist()}


def main():
    root = Path('results/thrml_spw/lfm3_1p7b')
    out = root / 'word_language'
    protocol = json.loads((root / 'word_language_protocol.json').read_text())
    state = json.loads((out / 'pipeline.json').read_text())
    assert state['status'] == 'all_five_word_seeds_independently_verified'
    assert state['completed_seeds'] == protocol['seeds']
    controls = json.loads((root / 'language_controls_batch/results.json').read_text())
    inputs = {}

    def load(path):
        inputs[str(path)] = sha(path)
        return json.loads(path.read_text())

    load(root / 'word_language_protocol.json')
    load(out / 'pipeline.json')
    load(root / 'language_controls_batch/results.json')
    load(root / 'language_controls_batch/independent_audit.json')
    load(root / 'language_native_v2/statistics_audit.json')
    nll = {'single_T1024': [], 'word4_T256': []}
    means = {key: [] for key in nll}
    seed_rows = []
    total_retained = total_single_warmup = total_word_warmup = 0
    for seed in protocol['seeds']:
        row = {'seed': seed}
        audits = {}
        for label, namespace in [('single_T1024', 'language_native_v2'), ('word4_T256', 'word_language')]:
            folder = root / namespace / str(seed)
            audit = load(folder / 'independent_audit.json')
            r = load(folder / 'results.json')
            assert audit['status'] == 'all_unique_parameters_and_saved_metrics_verified'
            assert audit['results_sha256'] == sha(folder / 'results.json')
            assert audit['unique_parameters_converted'] == 1728801536 and audit['seed'] == seed
            audits[label] = audit
            metrics = {m['stage']: m for m in r['metrics']}
            losses = np.asarray(metrics['native']['token_NLL'], dtype=np.float64)
            assert losses.shape == (8, 64) and np.isfinite(losses).all()
            nll[label].append(losses)
            means[label].append(float(losses.mean()))
            row[label] = {'NLL': float(losses.mean()), 'subset_perplexity': float(np.exp(losses.mean())),
                          'context_mean_NLL': losses.mean(1).tolist(),
                          'nominal_control_NLL': metrics['nominal_controls']['mean_NLL_FP64']}
        single, word = audits['single_T1024'], audits['word4_T256']
        assert single['retained_evaluations_including_padding'] == word['retained_evaluations_including_padding']
        assert word['warmup_evaluations_including_padding'] == 4 * single['warmup_evaluations_including_padding']
        assert word['retained_work_matches_paired_single_T1024'] and word['warmup_four_times_paired_single']
        row['paired_single_minus_word_NLL'] = means['single_T1024'][-1] - means['word4_T256'][-1]
        total_retained += single['retained_evaluations_including_padding']
        total_single_warmup += single['warmup_evaluations_including_padding']
        total_word_warmup += word['warmup_evaluations_including_padding']
        seed_rows.append(row)
    primary = contrast('single_T1024_minus_word4_T256', np.asarray(means['single_T1024']) - means['word4_T256'])
    secondary = []
    for label in nll:
        for baseline in controls['metrics']:
            base = np.asarray(baseline['token_NLL'], dtype=np.float64)
            assert base.shape == (8, 64)
            difference = (np.asarray(nll[label]) - base[None, :, :]).mean((1, 2))
            secondary.append(contrast(label + '_minus_' + baseline['condition'], difference))
    resources = load(root / 'language_resource_model/results.json')
    n = 1728801536
    metadata = resources['range_metadata_FP32_pair_bytes']
    summary = {'status': 'all_five_paired_seeds_primary_and_eight_control_contrasts_reported',
               'source_sha256': sha(Path(__file__)), 'inputs_sha256': inputs,
               'seed_count': 5, 'predicted_tokens_per_seed': 512, 'seeds': seed_rows,
               'primary_single_minus_word': primary, 'secondary_control_contrasts': secondary,
               'single_mean_NLL': float(np.mean(means['single_T1024'])),
               'word_mean_NLL': float(np.mean(means['word4_T256'])),
               'matched_retained_evaluations_per_method_cohort': total_retained,
               'single_warmup_evaluations_cohort': total_single_warmup,
               'word_warmup_evaluations_cohort': total_word_warmup,
               'fully_resident_logical_pbits': {'single': n, 'word': 4 * n},
               'conditional_packed_state_command_FP32range_bytes': {
                   'single_1state_12control_bits': n * 13 // 8 + metadata,
                   'word4_4state_4sign_bits': n + metadata},
               'conditional_single_over_word_sampling_latency_same_rate_full_parallelism': 4,
               'scope': 'All five paired inference seeds and all512 fixed tokens. Positive primary NLL difference favors word. MC t4 interval describes seed variation conditional on frozen text/checkpoint/settings, not unseen-text generalization or equivalence. Retained scheduled node work matched; warmup differs4times. Four-node format is internal positional baseline, not asserted Extropic hardware encoding. Packed payloads assume digital controls and omit physical area, framework/model/activation/compiler/IO overhead. No empirical energy, hardware latency or physical memory improvement claim.'}
    with (out / 'statistics.json').open('x') as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: summary[k] for k in ['status', 'single_mean_NLL', 'word_mean_NLL', 'primary_single_minus_word']}))


if __name__ == '__main__':
    main()
