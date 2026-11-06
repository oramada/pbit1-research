"""Independent two-outcome enumeration and scalar metric checks."""
import collections
import datetime
import json
import math

import numpy as np

from results.thrml_spw.fashion_cnn_native.probability_mechanism.study import ROOT, FINE, OUT, CONDITIONS, PREFIXES, sha, write, guard


def check_metrics(row, values):
    codes, probability, mean, variance, slope, target = [values[name] for name in ['codes', 'probability', 'mean', 'variance', 'slope', 'target']]
    n = len(codes)
    histogram = collections.Counter(int(x) for x in codes)
    assert row['weights'] == n and row['command_histogram'] == [histogram.get(k, 0) for k in range(256)]
    calculated = {'mean_single_sample_variance': sum(variance)/n,
                  'mean_squared_field_sensitivity': sum(float(x)**2 for x in slope)/n,
                  'endpoint_command_fraction': sum(int(x) in [0, 255] for x in codes)/n,
                  'probability_extreme_fraction_001': sum(x <= .01 or x >= .99 for x in probability)/n,
                  'mean_stationary_cell_entropy_bits': sum(-x*math.log2(x)-(1-x)*math.log2(1-x) for x in probability)/n,
                  'empirical_command_frequency_entropy_bits': -sum((count/n)*math.log2(count/n) for count in histogram.values()),
                  'mean_squared_change_from_starting_FP32': sum(float(x-y)**2 for x, y in zip(mean, target))/n}
    for name, value in calculated.items():
        np.testing.assert_allclose(value, row[name], rtol=5e-11, atol=2e-15)


def main():
    assert not (OUT/'completion.json').exists()
    p = json.loads((OUT/'protocol.json').read_text())
    guard(p)
    data = json.loads((OUT/'moments.json').read_text())
    assert data['protocol_sha256'] == sha(OUT/'protocol.json') and data['vectors_sha256'] == sha(OUT/'vectors.npz')
    vectors = np.load(OUT/'vectors.npz')
    originals = json.loads((ROOT/'training/manifest.json').read_text())['rows']
    fine = {(r['model'], r['method']): r for r in json.loads((FINE/'manifest.json').read_text())['rows']}
    layer_rows = {(r['model'], r['condition'], r['layer']): r for r in data['layers']}
    model_rows = {(r['model'], r['condition']): r for r in data['models']}
    checked = 0
    for mi in range(5):
        original = np.load(originals[mi]['path'])
        for ci, condition in enumerate(CONDITIONS):
            trained = None if ci == 0 else np.load(fine[mi, condition]['path'])
            joined = {name: [] for name in ['codes', 'probability', 'mean', 'variance', 'slope', 'target']}
            for li, prefix in enumerate(PREFIXES):
                w = original[prefix+'_weight'].astype(float).reshape(len(original[prefix+'_weight']), -1)
                mean, variance, slope, probability, codes = [], [], [], [], []
                for output, weights in enumerate(w):
                    lo, hi = min(weights), max(weights)
                    for wi, weight in enumerate(weights):
                        if trained is None:
                            q = min(1/(1+math.exp(-8.)), max(1/(1+math.exp(8.)), (weight-lo)/(hi-lo)))
                            code = round((math.log(q/(1-q))/2+4)*255/8)
                        else:
                            code = int(trained[prefix+'_commands'][output, wi])
                        field = float(np.float32(-4+code*8/255))
                        exp_field = math.exp(2*field)
                        prob = exp_field/(1+exp_field)
                        states, mass, z = np.array([lo, hi]), np.array([1-prob, prob]), np.array([0., 1.])
                        mu = float(np.sum(states*mass))
                        mean.append(mu)
                        variance.append(float(np.sum((states-mu)**2*mass)))
                        slope.append(float(2*np.sum(states*(z-prob)*mass)))
                        probability.append(prob)
                        codes.append(code)
                values = {'codes': np.array(codes, np.uint8), 'probability': np.array(probability), 'mean': np.array(mean),
                          'variance': np.array(variance), 'slope': np.array(slope), 'target': w.ravel()}
                for name, value in values.items():
                    saved = vectors[f'c{ci}_m{mi}_l{li}_{name}']
                    if name in ['codes', 'target']:
                        np.testing.assert_array_equal(value, saved)
                    else:
                        np.testing.assert_allclose(value, saved, rtol=5e-11, atol=2e-15)
                    joined[name].append(value)
                check_metrics(layer_rows[mi, condition, prefix], values)
                checked += len(codes)
            check_metrics(model_rows[mi, condition], {name: np.concatenate(value) for name, value in joined.items()})
    assert checked == 39960 and len(layer_rows) == 45 and len(model_rows) == 15
    for summary in data['summaries']:
        selected = [r for r in data['models'] if r['condition'] == summary['condition']]
        for name, value in summary.items():
            if name not in ['condition', 'model_units']:
                np.testing.assert_allclose(value, sum(row[name] for row in selected)/5, rtol=2e-13, atol=2e-15)
    write(OUT/'evidence_audit.json', {'passed': True, 'source_sha256': sha(__file__), 'weight_conditions_checked': checked,
                                   'layer_summaries_checked': 45, 'model_summaries_checked': 15,
                                   'scope': 'Separate scalar two-outcome law, covariance derivative and Counter-based code/entropy checks of every retained weight condition and summary. Nominal parameter diagnostic only, not native sampling or task-quality inference.', 'new_native_updates': 0})
    paths = [OUT/name for name in ['protocol.json', 'vectors.npz', 'moments.json', 'evidence_audit.json']]
    guard(p)
    write(OUT/'completion.json', {'status': 'all39960nominal_weight_conditions_checked_goal_active',
                                 'recorded_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                 'artifact_sha256': {str(path): sha(path) for path in paths}})
    print('All39960nominalweightconditions and layer/model summaries checked', flush=True)


if __name__ == '__main__':
    main()
