"""Serialize finite commands/fixed shared ranges without reconstructing weights."""
import math
from pathlib import Path
import numpy as np


def export_program(encoder, path):
    codes, _ = encoder.commands()
    with Path(path).open('xb') as f:
        np.savez_compressed(f, codes=codes.detach().cpu().numpy().astype(np.uint16),
            lower=encoder.lower.detach().cpu().numpy(), span=encoder.span.detach().cpu().numpy(),
            shape=np.asarray(encoder.original_shape, dtype=np.int64),
            group_size=np.asarray(encoder.group_size, dtype=np.int64),
            format_version=np.asarray(1, dtype=np.int64))


def load_program(path):
    with np.load(path) as a:
        arrays = {name: a[name].copy() for name in a.files}
    assert set(arrays) == {'codes', 'lower', 'span', 'shape', 'group_size', 'format_version'}
    shape = tuple(int(v) for v in arrays['shape'])
    assert len(shape) in [1, 2, 3] and int(arrays['format_version']) == 1
    assert arrays['codes'].shape == shape and arrays['codes'].dtype == np.uint16
    assert np.max(arrays['codes']) <= 4095
    rows = shape[0] if len(shape) > 1 else 1
    width = math.prod(shape[1:]) if len(shape) > 1 else shape[0]
    size = min(128, width)
    assert int(arrays['group_size']) == size and width % size == 0
    for name in ['lower', 'span']:
        assert arrays[name].dtype == np.float32
        assert arrays[name].shape == (rows, width // size, 1)
        assert np.isfinite(arrays[name]).all()
    assert np.min(arrays['span']) >= 0
    return arrays


def operator_layout(program, kind, rows=None):
    codes, lower, span = [program[k] for k in ['codes', 'lower', 'span']]
    size = int(program['group_size'])
    if kind == 'linear':
        assert codes.ndim == 2 and rows is None
        return {'codes': codes.T.copy(), 'lower': lower[:, :, 0].T.copy(), 'span': span[:, :, 0].T.copy()}
    if kind == 'conv':
        assert codes.ndim == 3 and codes.shape[1] == 1 and rows is None
        return {'codes': codes[:, 0, :].T.copy(), 'lower': lower[:, :, 0].T.copy(), 'span': span[:, :, 0].T.copy()}
    assert kind in ['embedding', 'elementwise']
    if kind == 'embedding':
        assert codes.ndim == 2
        if rows is not None:
            codes, lower, span = codes[rows], lower[rows], span[rows]
    else:
        assert codes.ndim == 1 and rows is None
    return {'codes': codes.reshape(1, -1).copy(),
            'lower': np.repeat(lower, size, axis=2).reshape(1, -1),
            'span': np.repeat(span, size, axis=2).reshape(1, -1)}
