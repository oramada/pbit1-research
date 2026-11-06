"""Lossless archives for float32 observations, separate from weight encoding.

Use 16-bit payloads only when every float32 low word is already zero. Otherwise
retain all32bits. No floating conversion/rounding is used in either path.
"""
import hashlib
import math
from pathlib import Path
import numpy as np


def data_digest(shape, values):
    h = hashlib.sha256(b'SPW_FLOAT32_OBSERVATION_V1')
    h.update(np.asarray([len(shape)], dtype='<i8').tobytes())
    h.update(np.asarray(shape, dtype='<i8').tobytes())
    h.update(values.tobytes(order='C'))
    return h.digest()


def write_logits(path, logits):
    array = np.asarray(logits)
    if array.dtype != np.dtype('<f4'):
        raise ValueError('Native little-endian float32 observations required')
    shape = array.shape
    canonical = np.ascontiguousarray(array).reshape(-1)
    bits = canonical.view('<u4')
    compact = bool(np.all((bits & 65535) == 0))
    payload = (bits >> 16).astype('<u2') if compact else canonical
    digest = data_digest(shape, canonical)
    path = Path(path)
    with path.open('xb') as handle:
        np.savez_compressed(handle, payload=payload, shape=np.asarray(shape, dtype='<i8'),
                            codec=np.asarray(int(compact), dtype=np.uint8),
                            data_sha256=np.frombuffer(digest, dtype=np.uint8),
                            format_version=np.asarray(1, dtype='<i8'))
    return {'codec': 'exact_high16' if compact else 'raw_float32',
            'shape': list(shape), 'raw_float32_bytes': canonical.nbytes,
            'payload_bytes': payload.nbytes, 'serialized_bytes': path.stat().st_size,
            'shape_and_data_sha256': digest.hex()}


def read_logits(path):
    with np.load(path, allow_pickle=False) as arrays:
        if set(arrays.files) != {'payload', 'shape', 'codec', 'data_sha256', 'format_version'}:
            raise ValueError('Unexpected archive fields')
        version, tag, dimensions = arrays['format_version'], arrays['codec'], arrays['shape']
        if version.dtype != np.dtype('<i8') or version.shape != () or int(version) != 1:
            raise ValueError('Unsupported observation format')
        if tag.dtype != np.uint8 or tag.shape != () or int(tag) not in (0, 1):
            raise ValueError('Unsupported payload codec')
        if dimensions.dtype != np.dtype('<i8') or dimensions.ndim != 1 or np.any(dimensions < 0):
            raise ValueError('Invalid shape')
        shape = tuple(int(v) for v in dimensions)
        payload, expected = arrays['payload'], arrays['data_sha256']
        if payload.ndim != 1 or math.prod(shape) != payload.size:
            raise ValueError('Shape and payload size disagree')
        if expected.dtype != np.uint8 or expected.shape != (32,):
            raise ValueError('Invalid integrity digest')
        if int(tag) == 1:
            if payload.dtype != np.dtype('<u2'):
                raise ValueError('Invalid compact payload type')
            values = (payload.astype('<u4') << 16).view('<f4')
        else:
            if payload.dtype != np.dtype('<f4'):
                raise ValueError('Invalid full-precision payload type')
            values = payload.copy()
    if data_digest(shape, values) != expected.tobytes():
        raise ValueError('Observation integrity check failed')
    return values.reshape(shape)
