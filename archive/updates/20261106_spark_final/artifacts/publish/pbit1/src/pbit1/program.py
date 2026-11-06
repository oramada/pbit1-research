"""Finite-control K1 research programs and independent-count reconstruction.

This is the NumPy R1 simulator used by the Spark research, not native THRML.
The controller stores uint16 commands restricted to 12 bits, plus FP32 ranges.
A sampled reconstruction is returned in FP32; neural BF16 rounding is external.
"""
from dataclasses import dataclass
import hashlib
import math
import numpy as np


@dataclass(frozen=True)
class Reconstruction:
    weights: np.ndarray
    counts_sha256: str
    counter_dtype: str
    maximum_count: int
    samples: int


@dataclass(frozen=True)
class FinitePBitProgram:
    shape: tuple[int, ...]
    group_size: int
    layout: str
    codes: np.ndarray
    lower: np.ndarray
    span: np.ndarray

    @classmethod
    def encode(cls, weights, *, group_size=128, layout="storage_flat"):
        if type(group_size) is not int or not 1 <= group_size <= 65536:
            raise ValueError("group_size must be an integer in [1, 65536]")
        w = np.asarray(weights, dtype=np.float32)
        if not w.size or not np.isfinite(w).all():
            raise ValueError("weights must be nonempty and finite")
        if layout not in ("storage_flat", "transposed_matrix"):
            raise ValueError("unknown layout")
        if layout == "transposed_matrix" and w.ndim != 2:
            raise ValueError("transposed_matrix requires a matrix")
        flat = (w.T if layout == "transposed_matrix" else w).reshape(-1)
        codes = np.empty(flat.size, dtype=np.uint16)
        n_groups = math.ceil(flat.size / group_size)
        lower = np.empty(n_groups, dtype=np.float32)
        span = np.empty(n_groups, dtype=np.float32)
        chunk = 262144 // group_size * group_size
        for start in range(0, flat.size, chunk):
            q = flat[start:start + chunk]
            blocks = np.pad(q, (0, (-q.size) % group_size), constant_values=np.nan).reshape(-1, group_size)
            lo = np.nanmin(blocks, axis=1)
            with np.errstate(over="ignore", invalid="ignore"):
                sc = np.nanmax(blocks, axis=1) - lo
            if not np.isfinite(sc).all():
                raise ValueError("group ranges must be representable in FP32")
            unit = np.divide(blocks.astype(np.float64) - lo[:, None], sc[:, None],
                             out=np.full(blocks.shape, .5, np.float64), where=sc[:, None] != 0).reshape(-1)[:q.size]
            edge = 1 / (1 + np.exp(8.))
            p = np.clip(unit, edge, 1 - edge)
            h = .5 * (np.log(p) - np.log1p(-p))
            codes[start:start + q.size] = np.rint(np.clip((h + 4) * (4095 / 8), 0, 4095)).astype(np.uint16)
            first = start // group_size
            lower[first:first + lo.size] = lo
            span[first:first + sc.size] = sc
        for a in (codes, lower, span):
            a.flags.writeable = False
        return cls(w.shape, group_size, layout, codes, lower, span)

    def _chunks(self):
        chunk = 262144 // self.group_size * self.group_size
        for start in range(0, self.codes.size, chunk):
            stop = min(start + chunk, self.codes.size)
            first, last = start // self.group_size, math.ceil(stop / self.group_size)
            fields = self.codes[start:stop].astype(np.float32) * np.float32(8 / 4095) - np.float32(4)
            p = 1 / (1 + np.exp(-2 * fields.astype(np.float64)))
            lo = np.repeat(self.lower[first:last], self.group_size)[:stop-start].astype(np.float64)
            sc = np.repeat(self.span[first:last], self.group_size)[:stop-start].astype(np.float64)
            yield start, stop, lo, sc, p

    def _restore(self, flat):
        oriented_shape = self.shape[::-1] if self.layout == "transposed_matrix" else self.shape
        a = flat.reshape(oriented_shape)
        return a.T.copy() if self.layout == "transposed_matrix" else a

    def expectation(self):
        flat = np.empty(self.codes.size, np.float32)
        for start, stop, lo, sc, p in self._chunks():
            flat[start:stop] = lo + sc * p
        return self._restore(flat)

    def reconstruct(self, n_samples, *, seed, parameter_index=0):
        if type(n_samples) is not int or not 1 <= n_samples <= 2**32 - 1:
            raise ValueError("n_samples must be an integer in [1, 2**32-1]")
        if type(parameter_index) is not int or parameter_index < 0:
            raise ValueError("parameter_index must be a nonnegative integer")
        rng = np.random.default_rng(np.random.SeedSequence([seed, parameter_index]))
        dtype = np.uint16 if n_samples <= 65535 else np.uint32
        flat = np.empty(self.codes.size, np.float32)
        digest, maximum = hashlib.sha256(), 0
        for start, stop, lo, sc, p in self._chunks():
            counts = rng.binomial(n_samples, p).astype(dtype)
            digest.update(counts.tobytes())
            maximum = max(maximum, int(counts.max()))
            flat[start:stop] = lo + sc * counts.astype(np.float64) / n_samples
        return Reconstruction(self._restore(flat), digest.hexdigest(), np.dtype(dtype).name,
                              maximum, n_samples)

    def storage(self):
        """Array payload only; excludes Python objects and reconstruction buffers."""
        n = self.codes.size
        metadata = self.lower.nbytes + self.span.nbytes
        return {"parameters": n, "logical_pbits": n,
                "controller_storage_bytes": self.codes.nbytes,
                "range_metadata_bytes": metadata,
                "actual_program_array_bytes": self.codes.nbytes + metadata,
                "ideal_packed_program_bits": 12 * n + 8 * metadata,
                "physical_binary_state_bits_additional": n}
