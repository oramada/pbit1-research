"""Check CUDA THRML execution against the preserved CPU operator before switching.

Backend rounding may change a few Bernoulli comparisons among hundreds of millions
of draws. Compare output differences with the actual stochastic error, not a false
promise of bitwise CPU/GPU identity. This is a simulator check, not TSU validation.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from .gpu_environment import cuda_environment


def main():
    block = Path("results/thrml_spw/transformer_block0")
    original = block/"runs/validation_T64_seed0"
    out = Path("results/thrml_spw/gpu_backend_validation");out.mkdir(parents=True, exist_ok=True)
    error_log = (out/"worker.log").open("w")
    worker = subprocess.Popen([str(Path('.venv-thrml-gpu/bin/python').absolute()), "-u", "-m", "research.thrml_spw.block_native_gpu_worker"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=error_log, text=True, env=cuda_environment())
    try:
        line = worker.stdout.readline()
        if not line:raise RuntimeError("GPU worker failed to initialize; inspect worker.log")
        ready = json.loads(line);assert ready["status"] == "ready"
        request = {"weights": str(block/"weights.npz"), "inputs": str(original/"stage0_inputs.npy"),
            "reference": str(original/"stage0_conditional_reference.npy"), "output": str(out/"gpu_stage0.npy"),
            "stage": 0, "samples": 64, "seed": 0, "domain": 940730, "contexts_per_batch": 8}
        worker.stdin.write(json.dumps(request)+"\n");worker.stdin.flush()
        response = json.loads(worker.stdout.readline())
        if response["status"] != "ok":raise RuntimeError(response)
        gpu, cpu, reference = [np.load(p).astype(np.float64) for p in
            (out/"gpu_stage0.npy", original/"stage0_native_output.npy", original/"stage0_conditional_reference.npy")]
        stochastic_rms = float(np.sqrt(np.mean((cpu-reference)**2)))
        backend_rms = float(np.sqrt(np.mean((gpu-cpu)**2)))
        ratio = backend_rms/stochastic_rms
        if ratio > .01:raise RuntimeError(f"Backend difference too large relative to stochastic noise: {ratio}")
        cpu_meta = json.loads((original/"stage0_native_output.json").read_text())
        gpu_meta = json.loads((out/"gpu_stage0.json").read_text())
        assert cpu_meta["free_node_updates"] == gpu_meta["free_node_updates"]
        assert cpu_meta["logical_weight_readouts"] == gpu_meta["logical_weight_readouts"]
        result = {"passed": True, "ready": ready, "cpu_stochastic_output_rms": stochastic_rms,
            "cpu_gpu_difference_rms": backend_rms, "difference_fraction_of_stochastic_rms": ratio,
            "max_absolute_backend_difference": float(np.max(np.abs(gpu-cpu))),
            "free_node_updates_per_backend": cpu_meta["free_node_updates"],
            "precision": gpu_meta["signature"]["matmul_precision"],
            "scope": "Same operator/inputs/keys and matched update/read counts; numerical simulator portability, not physical probabilistic hardware or a speed/energy claim",
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        (out/"REPORT.json").write_text(json.dumps(result, indent=2)+"\n")
        print(json.dumps(result, indent=2), flush=True)
    finally:
        if worker.poll() is None:
            worker.stdin.write('{"action":"quit"}\n');worker.stdin.flush();worker.wait(timeout=60)
        worker.stdin.close();worker.stdout.close();error_log.close()


if __name__ == "__main__":main()
