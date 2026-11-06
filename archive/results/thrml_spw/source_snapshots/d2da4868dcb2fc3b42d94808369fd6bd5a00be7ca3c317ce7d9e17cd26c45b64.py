"""Fixed conventions for same-fit, matched-retained-work word comparison."""
import json
import numpy as np
from results.thrml_spw.cifar_cnn_native.probability_cohort.common import (
    ROOT, METHODS, checkpoint, sha, write, save, guard, seal, metrics)

OUT = ROOT/'learned_word_cohort_evidence'
LOW = ROOT/'probability_cohort_evidence'
HIGH = ROOT/'probability_high_budget_evidence'
SAME = ROOT/'same_fit_bf16_evidence'
BASE = ROOT/'cohort_evidence'
WITNESS = ROOT/'learned_word_witness_v2_evidence'
CONTRASTS = ['single_minus_word', 'word_minus_nominal_word_mean', 'word_minus_samefit_BF16_CPU']


def closed():
    for folder in [LOW, HIGH, SAME, WITNESS]:
        seal(folder/'completion.json'); seal(folder/'review_completion.json')


def gpu_jobs():
    from pathlib import Path
    found = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            argv = (proc/'cmdline').read_bytes().split(b'\0')
            if b'.venv-thrml-gpu/bin/python' not in argv[0] or b'-m' not in argv:
                continue
            module = argv[argv.index(b'-m')+1]
            stat = (proc/'stat').read_text().rsplit(')', 1)[1].split()
            if module.startswith((b'results.thrml_spw.', b'research.thrml_spw.')) and stat[0] not in ['Z', 'X']:
                found.append({'pid': int(proc.name), 'start_ticks': stat[19], 'module': module.decode()})
        except (FileNotFoundError, ProcessLookupError, PermissionError, IndexError):
            continue
    return found


def keys(p, method, model, seed, samples, layer, first, count):
    import jax
    import jax.numpy as jnp
    key = jax.random.key(p['key_root'])
    for value in [method, model, seed, samples, layer]:
        key = jax.random.fold_in(key, value)
    return jax.vmap(lambda bank: jax.random.fold_in(key, bank))(
        jnp.arange(first, first+count, dtype=jnp.uint32))


def contrast(left, right, bm, left_seeds, right_seeds=None):
    lhs = left[bm[:, :, None], left_seeds].mean((1, 2))
    rhs = right[bm].mean(1) if right_seeds is None else right[bm[:, :, None], right_seeds].mean((1, 2))
    point = left.mean((0, 1))-(right.mean(0) if right_seeds is None else right.mean((0, 1)))
    return point, lhs-rhs


def singles():
    low = np.load(LOW/'statistical_arrays.npz'); high = np.load(HIGH/'statistical_arrays.npz')
    return (np.concatenate([low['values'], high['values'][:, None]], axis=1),
            [low['bootstrap_method_seeds']]*3+[high['bootstrap_method_seeds']])


def independent_mean(model):
    result = []
    for prefix in ['conv1', 'conv2', 'head']:
        w = model[prefix+'_weight']; c = model[prefix+'_center']; r = model[prefix+'_radius']
        flat = w.reshape(len(w), -1)
        q = np.divide(flat-c[:, None], r[:, None], out=np.zeros_like(flat), where=r[:, None] > 0)/2+.5
        code = np.round(np.minimum(1, np.maximum(0, q))*15).astype(np.uint8)
        expectation = np.zeros_like(flat)
        for bit in range(4):
            field = np.where((code >> bit) & 1, 4., -4.)
            expectation += (2**bit)/(1+np.exp(-2*field))
        result.append(((c-r)[:, None]+2*r[:, None]*expectation/15).reshape(w.shape))
    return result
