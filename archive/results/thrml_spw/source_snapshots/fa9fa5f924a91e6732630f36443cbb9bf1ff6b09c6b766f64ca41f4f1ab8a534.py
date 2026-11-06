"""Produce and audit layer calibration once; do not launch sampled inference."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    root = Path('results/thrml_spw/lfm3_1p7b')
    path = root / 'layer_calibration_pipeline.json'
    state = {'status': 'running', 'pid': os.getpid(), 'completed': [],
             'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    with path.open('x') as f:
        json.dump(state, f, indent=2)
    for module in ['calibrate_layer_allocation', 'verify_layer_calibration']:
        with (root / (module + '.log')).open('x') as log:
            child = subprocess.Popen([sys.executable, '-m', 'research.thrml_spw.lfm3.' + module], stdout=log, stderr=log)
            state.update(stage=module, child_pid=child.pid)
            path.write_text(json.dumps(state, indent=2) + '\n')
            code = child.wait()
        if code:
            state.update(status='failed_preserved_no_restart', child_exit_code=code)
            path.write_text(json.dumps(state, indent=2) + '\n')
            raise SystemExit(code)
        state['completed'].append(module)
    state.update(status='full_layer_proxy_and_allocation_candidate_verified_pending_sampled_quality', child_pid=None)
    path.write_text(json.dumps(state, indent=2) + '\n')


if __name__ == '__main__':
    main()
