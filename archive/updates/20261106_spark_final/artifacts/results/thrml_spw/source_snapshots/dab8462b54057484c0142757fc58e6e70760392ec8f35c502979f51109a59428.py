"""Fixed RGB matched-resource comparison conventions."""
import json
from pathlib import Path
from results.thrml_spw.cifar_cnn_native.cohort.common import ROOT, TRAIN, sha, write, save, guard
from results.thrml_spw.fashion_cnn_native.word_comparison.common import mean_weights

PARENT = ROOT/'cohort_evidence'
OUT = ROOT/'word_cohort_evidence'


def parent_closed():
    assert json.loads((PARENT/'pipeline.json').read_text())['status'] == 'all150compactRGBnative_cases_independently_audited_goal_active'
    for path, expected in json.loads((PARENT/'completion.json').read_text())['artifact_sha256'].items():
        assert sha(path) == expected, path
    assert json.loads((PARENT/'native_audit.json').read_text())['passed']
    assert json.loads((PARENT/'statistics_audit.json').read_text())['passed']


def keys(phase, model, seed, samples, layer, first, count):
    import jax
    import jax.numpy as jnp
    key = jax.random.key(1777520)
    for value in [phase, model, seed, samples, layer]:
        key = jax.random.fold_in(key, value)
    return jax.vmap(lambda bank: jax.random.fold_in(key, bank))(jnp.arange(first, first+count, dtype=jnp.uint32))
