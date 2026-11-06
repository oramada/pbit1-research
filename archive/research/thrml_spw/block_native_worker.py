"""Persistent local THRML worker for sequential multi-operator inference.

JSON messages carry local artifact paths, never reconstructed weights. Each
operator's immutable probability bank is cached across requests. Completed output
files are atomic and resumable only when inputs, weights, source and keys match.
"""
from __future__ import annotations

import hashlib
import json
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
from pathlib import Path
import resource
import sys
import time
import traceback

import jax
import jax.numpy as jnp
import numpy as np

from .provenance import manifest
from .range_native import RangeNativeLinear


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):h.update(block)
    return h.hexdigest()


def atomic_npy(path, array):
    path = Path(path);path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    with temporary.open("wb") as f:np.save(f, array)
    os.replace(temporary, path)


def atomic_json(path, value):
    path = Path(path);path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, indent=2)+"\n")
    os.replace(temporary, path)


class Worker:
    def __init__(self):
        self.banks = {}
        source = Path(__file__).parent
        self.source_hashes = {str(source/name): sha(source/name) for name in
            ("block_native_worker.py", "range_native.py", "native.py", "positive_inference.py")}

    def execute(self, request):
        for path, digest in self.source_hashes.items():
            if sha(path) != digest:raise RuntimeError(f"Live worker source changed: {path}")
        started = time.monotonic()
        master, inputs, reference, output = [Path(request[k]) for k in ("weights", "inputs", "reference", "output")]
        meta = output.with_suffix(".json")
        signature = {"weights_sha256": sha(master), "inputs_sha256": sha(inputs),
                     "reference_sha256": sha(reference), "stage": int(request["stage"]),
                     "samples": int(request["samples"]), "seed": int(request["seed"]),
                     "domain": int(request["domain"]), "contexts_per_batch": int(request["contexts_per_batch"]),
                     "source_sha256": self.source_hashes}
        if meta.exists() and output.exists():
            old = json.loads(meta.read_text())
            if old.get("status") == "complete" and old.get("signature") == signature and old.get("output_sha256") == sha(output):
                return {"status": "ok", "cached": True, "metadata": str(meta), "output": str(output)}
        cache_key = (str(master), signature["weights_sha256"], signature["stage"])
        if cache_key not in self.banks:
            z = np.load(master)
            i = signature["stage"]
            self.banks[cache_key] = RangeNativeLinear(z[f"weight_{i}"], z[f"bias_{i}"])
        layer = self.banks[cache_key]
        x, ref = np.load(inputs), np.load(reference)
        if x.ndim != 3 or x.shape[-1] != layer.bank.shape[0]:raise ValueError("Bad operator input shape")
        t, batch_size = signature["samples"], signature["contexts_per_batch"]
        if t < 1 or batch_size < 1:raise ValueError("Positive sample and batch counts required")
        chunks = [];fn = layer.compiled(t)
        for batch, first in enumerate(range(0, len(x), batch_size)):
            key = jax.random.key(signature["domain"])
            for value in (signature["seed"], signature["stage"], batch):key = jax.random.fold_in(key, value)
            xb = x[first:first+batch_size]
            y = fn(key, jnp.asarray(xb.reshape(-1, xb.shape[-1])))
            chunks.append(np.asarray(y).reshape(xb.shape[0], xb.shape[1], -1))
        y = np.concatenate(chunks)
        if y.shape != ref.shape:raise ValueError("Bad operator output shape")
        p = layer.bank.realized_probabilities
        variance = layer.span**2*p*(1-p)
        mean_x2 = np.mean(x.astype(np.float64)**2, axis=(0, 1))
        predicted = float(np.mean(mean_x2@variance)/t)
        actual = float(np.mean((y.astype(np.float64)-ref)**2))
        batches = (len(x)+batch_size-1)//batch_size
        result = {"status": "complete", "signature": signature, "controls": layer.controls(),
                  "input_shape": list(x.shape), "output_shape": list(y.shape),
                  "conditional_linear_mse": actual, "predicted_sampling_mse": predicted,
                  "free_node_updates": batches*t*len(layer.bank.free_indices),
                  "logical_weight_readouts": batches*t*layer.bank.logical_pbits,
                  "binary_product_terms": int(x.shape[0]*x.shape[1]*layer.bank.logical_pbits*t),
                  "bank_batches": batches, "elapsed_seconds": time.monotonic()-started,
                  "worker_peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024,
                  "scope": "Actual THRML linear-output accumulation on current student inputs; CPU/JAX arithmetic, no hardware energy/latency or weight-buffer compression claim"}
        atomic_npy(output, y)
        result["output_sha256"] = sha(output)
        # Preserve the normal official-source/package manifest for every completed
        # operator, and a small atomic checkpoint used for safe reuse.
        manifest(output.with_suffix(".manifest.json"), request, result)
        atomic_json(meta, result)
        return {"status": "ok", "cached": False, "metadata": str(meta), "output": str(output)}


def main():
    worker = Worker()
    print(json.dumps({"status": "ready", "backend": "official THRML/JAX"}), flush=True)
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if request.get("action") == "quit":break
            print(json.dumps(worker.execute(request)), flush=True)
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            print(json.dumps({"status": "error", "type": type(exc).__name__, "message": str(exc)}), flush=True)


if __name__ == "__main__":main()
