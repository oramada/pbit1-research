"""Frozen conventions for full native learned-probability CNN comparison."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path('results/thrml_spw/cifar_cnn_native')
OUT = ROOT/'probability_cohort_evidence'
PARENT = ROOT/'probability_native_witness_evidence'
BASE = ROOT/'cohort_evidence'
WORD = ROOT/'word_cohort_evidence'
FINE = ROOT/'probability_finetune_evidence'
METHODS = ['expected_weight', 'binomial_mean_STE_T256']
CONTRASTS = ['expected_minus_original', 'sampled_minus_original', 'sampled_minus_expected',
             'expected_minus_FP32', 'sampled_minus_FP32', 'expected_minus_own_mean', 'sampled_minus_own_mean',
             'expected_minus_INT4', 'sampled_minus_INT4', 'expected_minus_word', 'sampled_minus_word']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, obj):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2)+'\n')
    tmp.replace(path)


def save(path, **arrays):
    tmp = path.with_suffix('.tmp')
    with tmp.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    tmp.replace(path)


def guard(p):
    for mapping in [p['source_sha256'], p['input_sha256']]:
        for file, expected in mapping.items():
            assert sha(file) == expected, file


def seal(path):
    for file, expected in json.loads(Path(path).read_text())['artifact_sha256'].items():
        assert sha(file) == expected, file


def parent_closed():
    assert json.loads((PARENT/'completion.json').read_text())['status'] == 'RGBlearned_commands_CPU_native_validation_passed_goal_active'
    seal(PARENT/'completion.json')
    seal(PARENT/'review_completion.json')
    seal(FINE/'completion.json')
    assert json.loads((PARENT/'evidence_audit.json').read_text())['passed']
    expected = {BASE: 'all150compactRGBnative_cases_independently_audited_goal_active',
                WORD: 'all150RGBword_cases_independently_audited_goal_active'}
    for folder, status in expected.items():
        assert json.loads((folder/'pipeline.json').read_text())['status'] == status
        seal(folder/'completion.json')
        assert json.loads((folder/'native_audit.json').read_text())['passed']
        assert json.loads((folder/'statistics_audit.json').read_text())['passed']


def checkpoint(model, method):
    rows = json.loads((FINE/'manifest.json').read_text())['rows']
    matches = [row for row in rows if row['model'] == model and row['method'] == METHODS[method]]
    assert len(matches) == 1
    row = matches[0]
    assert sha(row['path']) == row['sha256']
    with np.load(row['path']) as data:
        return {name: data[name] for name in data.files}


def metrics(z, labels):
    shifted = z-z.max(1, keepdims=True)
    loss = np.log(np.exp(shifted).sum(1))-shifted[np.arange(len(labels)), labels]
    return np.array([np.mean(z.argmax(1) == labels), loss.mean()])


def keys(p, method, model, seed, t, layer, first, n):
    import jax
    import jax.numpy as jnp
    root = jax.random.key(p['key_root'])
    for item in [method, model, seed, t, layer]:
        root = jax.random.fold_in(root, item)
    return jax.vmap(lambda bank: jax.random.fold_in(root, bank))(jnp.arange(first, first+n, dtype=jnp.uint32))


def contrast_inputs(name, budget, values, base, word, means, base_metrics, controls):
    # Arrays are [model, MCseed, metric] or [model, metric].
    left = values[1 if name.startswith('sampled') else 0, budget]
    suffix = name.split('_minus_')[1]
    right = {'original': base[budget], 'expected': values[0, budget], 'FP32': base_metrics[controls.index('FP32_F64')],
             'own_mean': means[1 if name.startswith('sampled') else 0],
             'INT4': base_metrics[controls.index('ACT8_INT4_minmax')], 'word': word[budget]}[suffix]
    return left, right
