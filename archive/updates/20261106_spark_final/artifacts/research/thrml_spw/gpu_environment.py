"""Launch CUDA JAX against the already-installed model environment libraries.

No global CUDA/library installation is modified. Version checks stay enabled.
The library search path must be supplied before starting the child interpreter.
"""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


def cuda_environment():
    root = Path(__file__).parents[2]
    site = root/f".venv/lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"
    libraries = [site/"nvidia/cu13/lib", site/"nvidia/cudnn/lib"]
    if not (libraries[1]/"libcudnn.so.9").exists():
        raise RuntimeError("Recorded Torch cuDNN library path is unavailable")
    old = os.environ.get("LD_LIBRARY_PATH", "")
    paths = [str(p) for p in libraries]+([old] if old else [])
    return {**os.environ, "LD_LIBRARY_PATH": os.pathsep.join(paths), "JAX_PLATFORMS": "cuda",
            "JAX_DEFAULT_MATMUL_PRECISION": "highest", "XLA_PYTHON_CLIENT_PREALLOCATE": "false"}


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "--":args = args[1:]
    if not args:raise SystemExit("Supply a command after --")
    raise SystemExit(subprocess.call(args, env=cuda_environment()))
