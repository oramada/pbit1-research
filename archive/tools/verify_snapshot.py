"""Verify every file in this bounded public snapshot."""
from pathlib import Path
import hashlib,json
root=Path(__file__).resolve().parents[1]
m=json.loads((root/'FILES.sha256.json').read_text())
for name,digest in m.items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
print(f"Verified {len(m)} snapshot file hashes")
