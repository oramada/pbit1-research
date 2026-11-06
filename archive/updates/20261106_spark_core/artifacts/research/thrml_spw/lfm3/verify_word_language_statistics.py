"""Independent Decimal checks of every paired word/single contrast and cost."""
from decimal import Decimal, localcontext
import hashlib
import json
import math
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(actual, expected):
    assert math.isclose(actual, float(expected), rel_tol=2e-12, abs_tol=2e-12), (actual, expected)


def mean(values):
    return sum(values, Decimal(0)) / len(values)


def check_contrast(actual, expected, name):
    assert actual['name'] == name and len(expected) == 5
    for x, y in zip(actual['per_seed_NLL_difference'], expected, strict=True):
        close(x, y)
    center = mean(expected)
    se = (sum((v - center) ** 2 for v in expected) / Decimal(20)).sqrt()
    width = Decimal('2.7764451051977987') * se
    close(actual['mean_NLL_difference'], center)
    close(actual['MC_standard_error'], se)
    close(actual['perplexity_ratio_exp_mean_difference'], center.exp())
    for x, y in zip(actual['MC95_t4_NLL_difference_interval'], [center - width, center + width], strict=True):
        close(x, y)
    for x, y in zip(actual['MC95_perplexity_ratio_interval'], [center - width, center + width], strict=True):
        close(x, y.exp())


def main():
    root = Path('results/thrml_spw/lfm3_1p7b')
    out = root / 'word_language'
    path = out / 'statistics.json'
    s = json.loads(path.read_text())
    assert s['status'] == 'all_five_paired_seeds_primary_and_eight_control_contrasts_reported'
    assert s['source_sha256'] == sha(Path(__file__).with_name('word_language_statistics.py'))
    for name, digest in s['inputs_sha256'].items():
        assert sha(Path(name)) == digest
    protocol = json.loads((root / 'word_language_protocol.json').read_text())
    baseline = json.loads((root / 'language_controls_batch/results.json').read_text(), parse_float=Decimal)
    assert len(s['seeds']) == s['seed_count'] == 5 and s['predicted_tokens_per_seed'] == 512
    means = {'single_T1024': [], 'word4_T256': []}
    retained = single_warmup = word_warmup = 0
    with localcontext() as ctx:
        ctx.prec = 45
        for row, seed in zip(s['seeds'], protocol['seeds'], strict=True):
            assert row['seed'] == seed
            audits = {}
            for label, namespace in [('single_T1024', 'language_native_v2'), ('word4_T256', 'word_language')]:
                folder = root / namespace / str(seed)
                r = json.loads((folder / 'results.json').read_text(), parse_float=Decimal)
                a = json.loads((folder / 'independent_audit.json').read_text())
                assert a['status'] == 'all_unique_parameters_and_saved_metrics_verified'
                assert a['seed'] == seed and a['unique_parameters_converted'] == 1728801536
                assert a['results_sha256'] == sha(folder / 'results.json')
                audits[label] = a
                m = next(v for v in r['metrics'] if v['stage'] == 'native')
                nominal = next(v for v in r['metrics'] if v['stage'] == 'nominal_controls')
                tokens = [v for context in m['token_NLL'] for v in context]
                assert len(tokens) == 512
                ce = mean(tokens)
                means[label].append(ce)
                close(row[label]['NLL'], ce)
                close(row[label]['subset_perplexity'], ce.exp())
                close(row[label]['nominal_control_NLL'], mean([v for context in nominal['token_NLL'] for v in context]))
                for x, context in zip(row[label]['context_mean_NLL'], m['token_NLL'], strict=True):
                    close(x, mean(context))
            single, word = audits['single_T1024'], audits['word4_T256']
            assert single['retained_evaluations_including_padding'] == word['retained_evaluations_including_padding']
            assert word['warmup_evaluations_including_padding'] == 4 * single['warmup_evaluations_including_padding']
            retained += single['retained_evaluations_including_padding']
            single_warmup += single['warmup_evaluations_including_padding']
            word_warmup += word['warmup_evaluations_including_padding']
            close(row['paired_single_minus_word_NLL'], means['single_T1024'][-1] - means['word4_T256'][-1])
        close(s['single_mean_NLL'], mean(means['single_T1024']))
        close(s['word_mean_NLL'], mean(means['word4_T256']))
        check_contrast(s['primary_single_minus_word'], [a - b for a, b in zip(means['single_T1024'], means['word4_T256'], strict=True)], 'single_T1024_minus_word4_T256')
        expected_names = []
        for label in means:
            for control in baseline['metrics']:
                name = label + '_minus_' + control['condition']
                expected_names.append(name)
                base = mean([v for context in control['token_NLL'] for v in context])
                actual = next(v for v in s['secondary_control_contrasts'] if v['name'] == name)
                check_contrast(actual, [v - base for v in means[label]], name)
        assert [v['name'] for v in s['secondary_control_contrasts']] == expected_names and len(expected_names) == 8
    assert s['matched_retained_evaluations_per_method_cohort'] == retained
    assert s['single_warmup_evaluations_cohort'] == single_warmup
    assert s['word_warmup_evaluations_cohort'] == word_warmup == 4 * single_warmup
    n = 1728801536
    assert s['fully_resident_logical_pbits'] == {'single': n, 'word': 4 * n}
    payload = s['conditional_packed_state_command_FP32range_bytes']
    assert payload['single_1state_12control_bits'] * 8 == 13 * n + 64 * 13531430
    assert payload['word4_4state_4sign_bits'] * 8 == 8 * n + 64 * 13531430
    assert s['conditional_single_over_word_sampling_latency_same_rate_full_parallelism'] == 4
    audit = {'status': 'all_five_paired_seeds_nine_contrasts_and_resource_totals_verified',
             'source_sha256': sha(Path(__file__)), 'statistics_sha256': sha(path),
             'scope': 'Independent Decimal arithmetic on audited token losses; resource totals checked separately. Five MC seed clusters conditional on512fixedtokens, not unseen-text/equivalence/hardware inference.'}
    with (out / 'statistics_audit.json').open('x') as f:
        json.dump(audit, f, indent=2)
    print(json.dumps(audit))


if __name__ == '__main__':
    main()
