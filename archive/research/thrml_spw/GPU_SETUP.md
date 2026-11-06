# CUDA simulator environment

This environment accelerates official THRML sampling in software. It does not
provide access to physical Extropic hardware or certify hardware energy/latency.

Recorded host: Linux aarch64, NVIDIA GB10, 20 CPU affinity cores. The separate
`.venv-thrml-gpu` uses JAX/jaxlib/CUDA13 plugin/PJRT 0.12.0.1 and THRML 0.1.4 at
upstream commit `2d54cd8404df85bb65703d7e8b4e84e48c1b72a9`.
`requirements-thrml-gpu.lock.txt` records installed Python packages; local CUDA
libraries remain external dependencies. Installed Torch libraries supply CUDA
13.0.88 and cuDNN 9.24.0. The launcher sets their library paths before interpreter
startup, requests CUDA and highest matmul precision, and disables JAX preallocation.
CUDA version checks remain enabled. No global library installations were modified.

JAX documents the local CUDA installation option in its
[official installation guide](https://docs.jax.dev/en/latest/installation.html).
This host was provisioned with the CPU lock plus `jax[cuda13-local]==0.12.0.1`.
The exact local paths are in gpu_environment.py; other hosts must supply compatible
libraries and independently validate their backend.

```bash
.venv/bin/python -m research.thrml_spw.gpu_environment --   .venv-thrml-gpu/bin/python -m unittest   research.thrml_spw.test_native research.thrml_spw.test_range_native
.venv/bin/python -m research.thrml_spw.transformer_block_gpu --probe-worker
.venv/bin/python -m research.thrml_spw.transformer_block_gpu
```

Raw backend comparison: results/thrml_spw/gpu_backend_validation/REPORT.json.
Complete experiment: results/thrml_spw/transformer_block0_gpu/manifest.json.
Partial CPU reference: results/thrml_spw/transformer_block0/CPU_REFERENCE_STATUS.json.
