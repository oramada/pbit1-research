"""Parameter-only nominal probability/variance diagnostics, no task scores."""
import datetime
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path('results/thrml_spw/fashion_cnn_native')
FINE = ROOT/'probability_finetune_evidence'
OUT = ROOT/'probability_mechanism_evidence'
CONDITIONS = ['starting', 'expected_weight', 'binomial_mean_STE_T256']
PREFIXES = ['conv1', 'conv2', 'head']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')


def guard(p):
    for mapping in [p['source_sha256'], p['input_sha256']]:
        for path, h in mapping.items():
            assert sha(path) == h, path


def metrics(codes, probability, mean, variance, slope, target):
    histogram = np.bincount(codes, minlength=256)
    frequency = histogram[histogram > 0]/len(codes)
    return {'weights': len(codes), 'command_histogram': histogram.tolist(),
            'mean_single_sample_variance': float(variance.mean()),
            'mean_squared_field_sensitivity': float(np.mean(slope*slope)),
            'endpoint_command_fraction': float(np.mean((codes == 0) | (codes == 255))),
            'probability_extreme_fraction_001': float(np.mean((probability <= .01) | (probability >= .99))),
            'mean_stationary_cell_entropy_bits': float(np.mean(-probability*np.log2(probability)-(1-probability)*np.log2(1-probability))),
            'empirical_command_frequency_entropy_bits': float(-np.sum(frequency*np.log2(frequency))),
            'mean_squared_change_from_starting_FP32': float(np.mean((mean-target)**2))}


def prepare():
    OUT.mkdir(exist_ok=True)
    assert not (OUT/'protocol.json').exists()
    originals = json.loads((ROOT/'training/manifest.json').read_text())
    fine = json.loads((FINE/'manifest.json').read_text())
    assert fine['complete'] and json.loads((FINE/'evidence_audit.json').read_text())['passed']
    paths = [ROOT/'training/manifest.json', FINE/'protocol.json', FINE/'manifest.json', FINE/'evidence_audit.json', FINE/'preparation_completion.json']
    paths += [Path(row['path']) for row in originals['rows']+fine['rows']]
    p = {'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
         'source_sha256': {str(path): sha(path) for path in Path(__file__).parent.glob('*.py')},
         'input_sha256': {str(path): sha(path) for path in paths},
         'conditions': CONDITIONS, 'models': 5, 'layers': PREFIXES,
         'probability_extreme_cutoff': .01, 'field_grid_bits': 8, 'field_bound': 4.,
         'scope': 'Post-training parameter-only descriptive diagnostic on all5models/all3conditions. No images/labels/task metrics or sampling. Nominal independent Bernoulli variance and field derivative only, not downstream CNN quality or empirical device behavior. Fixed ranges are preserved; methods/budgets are not selected from this analysis. Cell entropy and empirical across-parameter code-frequency entropy are distinct quantities and neither proves a storage saving.'}
    guard(p)
    write(OUT/'protocol.json', p)


def run():
    p = json.loads((OUT/'protocol.json').read_text())
    guard(p)
    assert not (OUT/'moments.json').exists()
    originals = json.loads((ROOT/'training/manifest.json').read_text())['rows']
    fine = {(r['model'], r['method']): r for r in json.loads((FINE/'manifest.json').read_text())['rows']}
    layers, models, vectors = [], [], {}
    for mi in range(5):
        original = np.load(originals[mi]['path'])
        for ci, condition in enumerate(CONDITIONS):
            trained = None if ci == 0 else np.load(fine[mi, condition]['path'])
            all_vectors = {name: [] for name in ['codes', 'probability', 'mean', 'variance', 'slope', 'target']}
            for li, prefix in enumerate(PREFIXES):
                weight = original[prefix+'_weight'].astype(float)
                flat = weight.reshape(len(weight), -1)
                c, r = (flat.max(1)+flat.min(1))/2, (flat.max(1)-flat.min(1))/2
                assert np.all(r > 0)
                if ci == 0:
                    q = np.clip((flat-c[:, None])/(2*r[:, None])+.5, 1/(1+np.exp(8.)), 1/(1+np.exp(-8.)))
                    codes = np.rint((np.log(q/(1-q))/2+4)*255/8).astype(np.uint8)
                else:
                    np.testing.assert_array_equal(c, trained[prefix+'_center'])
                    np.testing.assert_array_equal(r, trained[prefix+'_radius'])
                    codes = trained[prefix+'_commands']
                    np.testing.assert_array_equal(codes, np.rint((trained[prefix+'_latent_fields']+4)*255/8).astype(np.uint8))
                fields = (-4+codes.astype(float)*8/255).astype(np.float32).astype(float)
                probability = 1/(1+np.exp(-2*fields))
                mean = c[:, None]-r[:, None]+2*r[:, None]*probability
                variance = 4*r[:, None]**2*probability*(1-probability)
                slope = 4*r[:, None]*probability*(1-probability)
                if trained is not None:
                    np.testing.assert_allclose(mean.reshape(weight.shape), trained[prefix+'_weight'], rtol=2e-14, atol=2e-15)
                values = {'codes': codes.ravel(), 'probability': probability.ravel(), 'mean': mean.ravel(),
                          'variance': variance.ravel(), 'slope': slope.ravel(), 'target': flat.ravel()}
                for name, value in values.items():
                    vectors[f'c{ci}_m{mi}_l{li}_{name}'] = value
                    all_vectors[name].append(value)
                layers.append({'model': mi, 'condition': condition, 'layer': prefix, **metrics(*[values[name] for name in all_vectors])})
            joined = {name: np.concatenate(values) for name, values in all_vectors.items()}
            models.append({'model': mi, 'condition': condition, **metrics(*[joined[name] for name in all_vectors])})
    assert len(layers) == 45 and len(models) == 15 and sum(r['weights'] for r in layers) == 39960
    summaries = []
    names = [key for key in models[0] if key not in ['model', 'condition', 'command_histogram', 'weights']]
    for condition in CONDITIONS:
        selected = [row for row in models if row['condition'] == condition]
        summaries.append({'condition': condition, 'model_units': 5,
                          **{key: float(np.mean([row[key] for row in selected])) for key in names}})
    np.savez_compressed(OUT/'vectors.npz', **vectors)
    guard(p)
    write(OUT/'moments.json', {'protocol_sha256': sha(OUT/'protocol.json'), 'vectors_sha256': sha(OUT/'vectors.npz'),
                             'layers': layers, 'models': models, 'summaries': summaries,
                             'scope': p['scope'], 'new_native_updates': 0})
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == '__main__':
    {'prepare': prepare, 'run': run}[sys.argv[1]]()
