"""Three-method, five-seed frozen LFM3 comparison at equal retained work."""
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summary(values):
    values = np.asarray(values, dtype=np.float64)
    assert values.shape == (5,)
    center = float(values.mean())
    se = float(values.std(ddof=1) / np.sqrt(5))
    half = 2.7764451051977987 * se
    return {'per_seed_NLL': values.tolist(), 'mean_NLL': center, 'MC_standard_error': se,
            'MC95_t4_interval': [center - half, center + half],
            'exp_mean_NLL': float(np.exp(center)), 'MC95_exp_interval': np.exp([center - half, center + half]).tolist()}


def main():
    root = Path('results/thrml_spw/lfm3_1p7b')
    out = root / 'word8_language_statistics_v2'
    out.mkdir(exist_ok=False)
    observation = (root / 'word8_language/pipeline.json').read_bytes()
    with (out / 'pipeline_observation.json').open('xb') as handle:
        handle.write(observation)
    inputs = {}
    def read(path):
        value = json.loads(path.read_text())
        inputs[str(path)] = sha(path)
        return value
    protocol = read(root / 'word8_language_protocol.json')
    assert read(out / 'pipeline_observation.json')['completed_seeds'] == protocol['seeds']
    methods = {'single_T1024': ('language_native_v2', 1, 1024),
               'word4_T256': ('word_language', 4, 256), 'word8_T128': ('word8_language', 8, 128)}
    values, resources, records = {}, {}, []
    for name, (namespace, bits, samples) in methods.items():
        nll, expected = [], []
        retained = warmup = 0
        for seed in protocol['seeds']:
            folder = root / namespace / str(seed)
            audit = read(folder / 'independent_audit.json')
            result = read(folder / 'results.json')
            assert audit['status'] == 'all_unique_parameters_and_saved_metrics_verified'
            assert audit['results_sha256'] == sha(folder / 'results.json')
            assert audit['samples'] == samples and audit['seed'] == seed and audit['unique_parameters_converted'] == 1728801536
            metrics = {v['stage']: v for v in result['metrics']}
            tokens = np.asarray(metrics['native']['token_NLL'], dtype=np.float64)
            assert tokens.shape == (8, 64) and np.isfinite(tokens).all()
            nll.append(float(tokens.mean())); expected.append(metrics['nominal_controls']['mean_NLL_FP64'])
            retained += audit['retained_evaluations_including_padding']
            warmup += audit['warmup_evaluations_including_padding']
            records.append({'method': name, 'seed': seed, 'native_NLL': nll[-1],
                            'retained': audit['retained_evaluations_including_padding'],
                            'warmup': audit['warmup_evaluations_including_padding']})
        assert len(set(expected)) == 1
        values[name] = summary(nll)
        values[name]['nominal_expected_operator_NLL'] = expected[0]
        n = 1728801536
        control_bits = 12 if bits == 1 else bits
        resources[name] = {'logical_pbits_if_fully_resident': bits * n, 'pbits_per_parameter': bits,
            'samples': samples, 'retained_evaluations_cohort': retained, 'warmup_evaluations_cohort': warmup,
            'programming_levels_per_node': 4096 if bits == 1 else 2,
            'conditional_packed_state_command_and_FP32ranges_bytes': (n * (bits + control_bits) + 64 * 13531430) // 8}
    assert len({v['retained_evaluations_cohort'] for v in resources.values()}) == 1
    assert resources['word8_T128']['warmup_evaluations_cohort'] == 8 * resources['single_T1024']['warmup_evaluations_cohort']
    assert resources['word8_T128']['warmup_evaluations_cohort'] == 2 * resources['word4_T256']['warmup_evaluations_cohort']
    pairwise = {}
    for first, second in itertools.combinations(methods, 2):
        pairwise[first + '_minus_' + second] = summary(np.asarray(values[first]['per_seed_NLL']) - values[second]['per_seed_NLL'])
    control = read(root / 'language_controls_batch/results.json')
    read(root / 'language_controls_batch/independent_audit.json')
    contrasts = {}
    for name in methods:
        for rec in control['metrics']:
            baseline = np.asarray(rec['token_NLL'], dtype=np.float64)
            assert baseline.shape == (8, 64)
            contrasts[name + '_minus_' + rec['condition']] = summary(np.asarray(values[name]['per_seed_NLL']) - float(baseline.mean()))
    result = {'status': 'all_five_K8_seeds_and_fifteen_three_method_contrasts_reported',
        'source_sha256': sha(Path(__file__)), 'inputs_sha256': inputs, 'methods': values,
        'resources': resources, 'seed_records': records, 'pairwise_NLL_differences': pairwise,
        'deterministic_control_contrasts': contrasts, 'primary_contrast': 'single_T1024_minus_word8_T128',
        'scope': 'Recovery analysis: immutable archived pipeline observation replaces the live state-file hash; no sampler or model repeated. Three frozen representations, five paired MC clusters and512fixedtargets. Equal retained schedulednodeevaluations withwarmup1:4:8. K8 is an internal positional256-level word withtwofieldlevels/node, not claimedvendor format. All15contrasts retained; intervals are conditional approximateMCt4 intervals, not unseen-data/training uncertainty, equivalence or multiplicity-controlled significance. Deviceassignments/payloads are conditional models; no measuredhardware energy/latency/area advantage. Nativewalltimes may reflect concurrent independent researchjobs and are not compared.'}
    with (out / 'statistics.json').open('x') as f:
        json.dump(result, f, indent=2)
    print(json.dumps({k: v['mean_NLL'] for k, v in values.items()}))


if __name__ == '__main__':
    main()
