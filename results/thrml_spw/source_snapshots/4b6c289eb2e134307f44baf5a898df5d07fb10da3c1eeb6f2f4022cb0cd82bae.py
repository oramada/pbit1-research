"""Cross-environment seed-index identity prerequisite; no task/native calls."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np


def main():
    path = Path(sys.argv[1])
    assert not path.exists()
    values = np.random.default_rng(1877520).integers(0, 10, size=(2, 5000, 5, 10))
    assert values.dtype == np.int64
    path.write_text(json.dumps({'numpy_version': np.__version__, 'seed': 1877520,
        'shape': list(values.shape), 'dtype': values.dtype.str,
        'indices_sha256': hashlib.sha256(values.tobytes()).hexdigest(),
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, indent=2)+'\n')


if __name__ == '__main__':
    main()
