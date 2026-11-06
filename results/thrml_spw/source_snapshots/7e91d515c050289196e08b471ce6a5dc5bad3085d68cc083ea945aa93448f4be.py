from __future__ import annotations

import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import subprocess
import sys
from datetime import datetime, timezone


def manifest(destination: Path, config: dict, result: dict):
    commit = subprocess.check_output(["git", "-C", "vendor/thrml", "rev-parse", "HEAD"], text=True).strip()
    source = Path(__file__).parent
    content = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "official_repository": "https://github.com/extropic-ai/thrml",
        "source_commit": commit,
        "python": sys.version, "platform": platform.platform(),
        "packages": {name: version(name) for name in ("thrml", "jax", "jaxlib", "equinox", "numpy", "matplotlib")},
        "script_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(source.glob("*.py"))},
        "config": config, "result": result,
        "scope": "Official THRML JAX sampling simulation; no physical-chip performance measurement",
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(content, indent=2) + "\n")
