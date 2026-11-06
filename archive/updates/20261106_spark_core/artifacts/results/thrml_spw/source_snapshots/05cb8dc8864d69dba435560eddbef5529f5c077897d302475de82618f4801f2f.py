"""Validation Gaussian sensitivity proxy and exact integer sampling allocation."""
from pathlib import Path
import json
import hashlib
import numpy as np
from results.thrml_spw.fashion_cnn_native.gaussian_control.common import moments, torch_forward
from results.thrml_spw.fashion_cnn_native.gaussian_control.preflight import numpy_forward

ROOT = Path('results/thrml_spw/fashion_cnn_native')
TRAIN = ROOT/'training'
OUT = ROOT/'layer_allocation_evidence'
PREFIXES = ['conv1', 'conv2', 'head']
COUNTS = np.array([72, 1152, 1440], np.int64)
KEY_ROOT = 1697520


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2)+'\n')
    temp.replace(path)


def guard(p):
    for mapping in [p['source_sha256'], p['input_sha256']]:
        for path, h in mapping.items():
            assert sha(path) == h, path


def weights_for(model, mi, seed, selected_layer, first, count):
    result = []
    for li, prefix in enumerate(PREFIXES):
        nominal, variance, _, _ = moments(model, prefix)
        if li != selected_layer:
            result.append(np.broadcast_to(nominal, (count, *nominal.shape)).copy())
        else:
            weights = []
            for bank in range(first, first+count):
                rng = np.random.default_rng(np.random.SeedSequence([KEY_ROOT, mi, seed, selected_layer, bank]))
                weights.append(nominal+np.sqrt(variance/256)*rng.standard_normal(nominal.shape))
            result.append(np.stack(weights))
    return result


def losses(logits, labels):
    maximum = logits.max(1)
    return maximum+np.log(np.exp(logits-maximum[:, None]).sum(1))-logits[np.arange(len(labels)), labels]


def solve(coefficients):
    # Divide weights by72: T0+16*T1+20*T2=9472 is exact uniformT256 cost.
    second, third = np.meshgrid(np.arange(64, 1025), np.arange(64, 1025), indexing='ij')
    first = 9472-16*second-20*third
    valid = (first >= 64) & (first <= 1024)
    schedules = np.stack([first[valid], second[valid], third[valid]], axis=1)
    scores = (coefficients[None, :]*256/schedules).sum(1)
    # Objective first, then explicit lexicographic tie-break across layer budgets.
    order = np.lexsort((schedules[:, 2], schedules[:, 1], schedules[:, 0], scores))
    index = order[0]
    chosen = schedules[index]
    assert int(chosen@COUNTS) == 2664*256
    return chosen, float(scores[index]), len(scores)
