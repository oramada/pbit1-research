"""Pure NumPy conventions and gates for the complete compact RGB cohort."""
from pathlib import Path
import json
from results.thrml_spw.fashion_cnn_native.cohort.common import (
    CONTROLS, sha, write, save, guard, control_weights, numpy_control, metrics)

ROOT = Path('results/thrml_spw/cifar_cnn_native')
TRAIN = ROOT/'training'
OUT = ROOT/'cohort_evidence'


def preparation_closed():
    for path, expected in json.loads((ROOT/'preparation_completion.json').read_text())['artifact_sha256'].items():
        assert sha(path) == expected, path
    assert json.loads((TRAIN/'evidence_audit.json').read_text())['passed']
    assert json.loads((ROOT/'native_witness_evidence/evidence_audit.json').read_text())['passed']


def key_bank(phase, model, seed, samples, layer, first, count):
    import jax
    import jax.numpy as jnp
    key = jax.random.key(1737520)
    for value in [phase, model, seed, samples, layer]:
        key = jax.random.fold_in(key, value)
    return jax.vmap(lambda bank: jax.random.fold_in(key, bank))(jnp.arange(first, first+count, dtype=jnp.uint32))
