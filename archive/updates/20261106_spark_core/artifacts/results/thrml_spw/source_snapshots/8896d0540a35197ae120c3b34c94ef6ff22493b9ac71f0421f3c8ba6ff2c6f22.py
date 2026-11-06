"""Full-checkpoint scalar moments for single versus all K=2/4/8 words.

Analytical marginal laws only; excludes activation quantization, nonlinear
propagation, cross-device correlation and hardware costs.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from safetensors import safe_open


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(16 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    torch.set_num_threads(2)
    root = Path('results/thrml_spw/lfm3_1p7b')
    out = root / 'spatial_moments'
    out.mkdir(exist_ok=True)
    checkpoint = Path('data/thrml_spw/lfm3_1p7b/checkpoint/model.safetensors')
    inventory = json.loads((root / 'meta_inventory.json').read_text())
    expected_sha = json.loads((root / 'preparation_inventory.json').read_text())['checkpoint_files']['model.safetensors']
    assert sha(checkpoint) == expected_sha
    methods = ['single12'] + [f'word{k}_endpoint' for k in [2,4,8]]
    protocol = {'source_sha256': sha(Path(__file__)), 'inventory_sha256': sha(root / 'meta_inventory.json'),
                'checkpoint_sha256': expected_sha, 'methods': methods,
                'range_group_size': 128, 'field_bounds': [-4, 4], 'single_control_bits': 12,
                'word_bits': [2,4,8], 'word_coefficients': {str(k): [2**bit/((1<<k)-1) for bit in range(k)] for k in [2,4,8]},
                'matched_evaluations_per_weight': [8,16,64,256,1024,4096,16384,65536,262144,1048576],
                'scope': 'Analytical marginal scalar MSE on all unique BF16 checkpoint weights using current block128 ranges. Single12-bit fields vs K independent finite-endpoint p-bits representing nearest INT2/4/8 word. Equal retained stochastic evaluations per scalar: singleT=C, wordT=C/K. Excludes warmup, activation quantization, nonlinear propagation, empirical correlation/calibration, hardware energy and task accuracy. Tied embedding/head counted once.'}
    with (out / 'protocol.json').open('x') as f:
        json.dump(protocol, f, indent=2)
    totals = {m: {'weights': 0, 'groups': 0, 'squared_bias_sum': 0., 'one_realization_variance_sum': 0.} for m in methods}
    layers = []
    delta = 1 / (1 + np.exp(8.0))
    with safe_open(str(checkpoint), framework='pt') as f:
        for parameter in inventory['parameters']:
            name, shape = parameter['name'], parameter['shape']
            vector = len(shape) == 1
            width = shape[0] if vector else int(np.prod(shape[1:]))
            rows = 1 if vector else shape[0]
            size = min(128, width)
            assert width % size == 0
            layer = {m: {k: 0 for k in totals[m]} for m in methods}
            for first in range(0, rows, 256):
                tensor = f.get_tensor(name) if vector else f.get_slice(name)[first:min(first + 256, rows)]
                w = tensor.float().numpy().astype(np.float64).reshape(-1, width // size, size)
                lower = w.min(2, keepdims=True)
                span = np.ptp(w, axis=2, keepdims=True)
                p = np.divide(w - lower, span, out=np.full_like(w, .5), where=span != 0).clip(0, 1)
                clipped = p.clip(1e-12, 1 - 1e-12)
                fields = ((np.log(clipped) - np.log1p(-clipped)) / 2).clip(-4, 4)
                code = np.rint((fields + 4) * 4095 / 8).astype(np.uint16)
                decoded = code.astype(np.float32) * np.float32(8 / 4095) - np.float32(4)
                q = np.exp(-np.logaddexp(0, -2 * decoded.astype(np.float64)))
                mean_single = lower + span * q
                quantities = {'single12': (mean_single, span * span * q * (1-q))}
                for bits in protocol['word_bits']:
                    maximum=(1<<bits)-1
                    word=np.rint(p*maximum).astype(np.uint16)
                    mean_word=lower+span*(delta+(1-2*delta)*word/maximum)
                    coefficient_squares=sum(4**bit for bit in range(bits))/maximum**2
                    variance=np.broadcast_to(span*span*delta*(1-delta)*coefficient_squares,w.shape)
                    quantities[f'word{bits}_endpoint']=(mean_word,variance)
                for method, (expected, variance) in quantities.items():
                    values = {'weights': w.size, 'groups': lower.size,
                              'squared_bias_sum': float(np.sum((expected - w) ** 2)),
                              'one_realization_variance_sum': float(np.sum(variance))}
                    for key, value in values.items():
                        layer[method][key] += value
                        totals[method][key] += value
            layers.append({'name': name, 'shape': shape, 'moments': layer})
            (out / 'progress.json').write_text(json.dumps({'completed_tensors': len(layers), 'last_tensor': name}) + '\n')
    for method, values in totals.items():
        assert values['weights'] == 1728801536 and values['groups'] == 13531430
        values['mean_squared_bias'] = values['squared_bias_sum'] / values['weights']
        values['mean_one_realization_variance'] = values['one_realization_variance_sum'] / values['weights']
        nodes = 1 if method == 'single12' else int(method[4])
        values['scalar_MSE_at_matched_evaluations'] = {
            str(c): values['mean_squared_bias'] + nodes * values['mean_one_realization_variance'] / c
            for c in protocol['matched_evaluations_per_weight']}
    single=totals['single12']
    crossings={}
    for bits in protocol['word_bits']:
        word=totals[f'word{bits}_endpoint']
        denominator=word['mean_squared_bias']-single['mean_squared_bias']
        crossings[str(bits)]=(single['mean_one_realization_variance']-bits*word['mean_one_realization_variance'])/denominator if denominator!=0 else None
    result = {'source_sha256': protocol['source_sha256'], 'summary': totals, 'layers': layers,
              'formal_scalar_MSE_crossing_evaluations_per_weight_by_K': crossings,
              'crossing_scope': 'Algebraic equality of aggregate marginal MSE curves; positive real crossing only meaningful within independent stationary model. Does not predict a neural quality crossover.',
              'scope': protocol['scope']}
    with (out / 'results.json').open('x') as f:
        json.dump(result, f, indent=2)
    print(json.dumps({'summary': totals, 'scalar_crossings': crossings}))


if __name__ == '__main__':
    main()
