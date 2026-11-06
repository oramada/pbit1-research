"""Independent Decimal arithmetic audit of all paired MC seed contrasts."""
from decimal import Decimal, localcontext
import hashlib
import json
from pathlib import Path
import math


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(actual, expected):
    assert math.isclose(actual, float(expected), rel_tol=2e-12, abs_tol=2e-12), (actual, expected)


def mean(values):
    return sum(values, Decimal(0)) / len(values)


def main():
    root = Path('results/thrml_spw/lfm3_1p7b')
    out = root / 'language_native_v2'
    summary_path = out / 'statistics.json'
    summary = json.loads(summary_path.read_text())
    assert summary['status'] == 'all_five_seeds_all_four_controls_reported'
    assert sha(Path(__file__).with_name('language_statistics.py')) == summary['source_sha256']
    for path, digest in summary['inputs_sha256'].items():
        assert sha(Path(path)) == digest
    protocol = json.loads((root / 'language_native_v2_protocol.json').read_text())
    controls = json.loads((root / 'language_controls_batch/results.json').read_text(), parse_float=Decimal)
    with localcontext() as ctx:
        ctx.prec = 45
        native = []
        retained = 0
        warmup = 0
        for row, seed in zip(summary['seeds'], protocol['seeds'], strict=True):
            assert row['seed'] == seed
            run = out / str(seed)
            r = json.loads((run / 'results.json').read_text(), parse_float=Decimal)
            a = json.loads((run / 'independent_audit.json').read_text())
            assert a['status'] == 'all_unique_parameters_and_saved_metrics_verified'
            assert a['unique_parameters_converted'] == 1728801536
            m = next(v for v in r['metrics'] if v['stage'] == 'native')
            nominal = next(v for v in r['metrics'] if v['stage'] == 'nominal_controls')
            close(summary['nominal_control_NLL'], mean([v for context in nominal['token_NLL'] for v in context]))
            tokens = [v for context in m['token_NLL'] for v in context]
            assert len(tokens) == 512
            ce = mean(tokens)
            native.append(ce)
            close(row['NLL'], ce)
            close(row['subset_perplexity'], ce.exp())
            for observed, context in zip(row['context_mean_NLL'], m['token_NLL'], strict=True):
                close(observed, mean(context))
            assert a['retained_evaluations_including_padding'] == summary['per_seed_retained_node_evaluations']
            assert a['warmup_evaluations_including_padding'] == summary['per_seed_warmup_node_evaluations']
            retained += a['retained_evaluations_including_padding']
            warmup += a['warmup_evaluations_including_padding']
        assert len(native) == summary['seed_count'] == 5
        assert summary['predicted_tokens_per_seed'] == 512
        assert summary['cohort_retained_node_evaluations'] == retained
        assert summary['cohort_warmup_node_evaluations'] == warmup
        average = mean(native)
        se = (sum((x - average) ** 2 for x in native) / Decimal(20)).sqrt()
        critical = Decimal('2.7764451051977987')
        close(summary['native_mean_NLL'], average)
        close(summary['native_MC_standard_error'], se)
        close(summary['native_subset_perplexity_exp_mean_NLL'], average.exp())
        for observed, expected in zip(summary['native_MC95_t4_NLL_interval'], [average - critical * se, average + critical * se], strict=True):
            close(observed, expected)
        assert len(summary['contrasts']) == len(controls['metrics']) == 4
        for contrast, baseline in zip(summary['contrasts'], controls['metrics'], strict=True):
            assert contrast['control'] == baseline['condition']
            base = mean([v for context in baseline['token_NLL'] for v in context])
            close(contrast['control_NLL'], base)
            differences = [v - base for v in native]
            for observed, expected in zip(contrast['per_seed_paired_NLL_difference'], differences, strict=True):
                close(observed, expected)
            delta = mean(differences)
            delta_se = (sum((v - delta) ** 2 for v in differences) / Decimal(20)).sqrt()
            close(contrast['mean_paired_NLL_difference'], delta)
            close(contrast['MC_standard_error'], delta_se)
            close(contrast['perplexity_ratio_exp_mean_NLL_difference'], delta.exp())
            interval = [delta - critical * delta_se, delta + critical * delta_se]
            for observed, expected in zip(contrast['MC95_t4_NLL_difference_interval'], interval, strict=True):
                close(observed, expected)
            for observed, expected in zip(contrast['MC95_perplexity_ratio_interval'], interval, strict=True):
                close(observed, expected.exp())
    audit = {'status': 'all_five_seed_statistics_and_four_contrasts_verified_by_decimal_arithmetic',
             'source_sha256': sha(Path(__file__)), 'statistics_sha256': sha(summary_path),
             'scope': 'MC statistics conditional on fixed512 tokens; saved logits and per-seed coverage audited upstream. No equivalence or unseen-text generalization inference.'}
    with (out / 'statistics_audit.json').open('x') as f:
        json.dump(audit, f, indent=2)
    print(json.dumps(audit))


if __name__ == '__main__':
    main()
