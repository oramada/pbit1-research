"""Variable-rank operator layout, including per-head attention normalization.

Elementwise input must already contain the original model's normalized
activation. Normalization/nonlinearities and dtype conversion belong to the
future inference runner, not this NumPy shape component.
"""
import numpy as np
from .program_export import operator_layout


def prepare(kind, inputs, program):
    x = np.asarray(inputs)
    shape = tuple(int(v) for v in program['shape'])
    state = {'kind': kind, 'input_shape': list(x.shape), 'weight_shape': list(shape)}
    if kind == 'embedding':
        if x.dtype.kind not in 'iu' or x.ndim != 2 or x.size == 0:
            raise ValueError('Embedding inputs must be a nonempty integer matrix')
        if np.min(x) < 0 or np.max(x) >= shape[0]:
            raise ValueError('Embedding index outside table')
        rows, inverse = np.unique(x, return_inverse=True)
        layout = operator_layout(program, kind, rows=rows)
        xx = np.ones((1, 1, layout['codes'].size), dtype=np.float32)
        state.update(rows=rows, inverse=inverse)
    else:
        if x.dtype != np.float32 or x.ndim < 1 or not np.isfinite(x).all() or x.size == 0:
            raise ValueError('Activation inputs must be finite nonempty float32 tensors')
        layout = operator_layout(program, kind)
        if kind == 'conv':
            if x.ndim != 3 or len(shape) != 3 or shape[1:] != (1, 7) or x.shape[1] != shape[0]:
                raise ValueError('Expected width-seven depthwise convolution inputs')
            windows = np.lib.stride_tricks.sliding_window_view(
                np.pad(x, ((0, 0), (0, 0), (6, 0))), 7, axis=2)
            xx = windows.transpose(0, 2, 3, 1).reshape(-1, 7, x.shape[1])
        elif kind == 'linear':
            if x.ndim < 2 or len(shape) != 2 or x.shape[-1] != shape[1]:
                raise ValueError('Linear input width mismatch')
            xx = x.reshape(-1, x.shape[-1])
        elif kind == 'elementwise':
            if len(shape) != 1 or x.shape[-1] != shape[0]:
                raise ValueError('Elementwise input width mismatch')
            xx = x.reshape(-1, 1, x.shape[-1])
        else:
            raise ValueError('Unsupported operator')
    state['worker_input_shape'] = list(xx.shape)
    state['worker_output_shape'] = [xx.shape[0], layout['codes'].shape[1]]
    return xx, layout, state


def restore(output, state):
    y = np.asarray(output)
    if list(y.shape) != state['worker_output_shape'] or not np.isfinite(y).all():
        raise ValueError('Worker output shape mismatch or nonfinite output')
    kind, xshape, wshape = state['kind'], state['input_shape'], state['weight_shape']
    if kind == 'embedding':
        return y.reshape(len(state['rows']), wshape[1])[state['inverse']].reshape(*xshape, wshape[1])
    if kind == 'conv':
        return np.pad(y.reshape(xshape[0], xshape[2], xshape[1]).transpose(0, 2, 1),
                      ((0, 0), (0, 0), (0, 6)))
    return y.reshape(*xshape[:-1], wshape[0])
