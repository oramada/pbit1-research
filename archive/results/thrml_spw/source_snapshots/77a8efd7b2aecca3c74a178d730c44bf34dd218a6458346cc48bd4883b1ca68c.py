"""Validation production, independent replay, allocation and independent optimum."""
import datetime
import json
import os
import subprocess
import sys
from results.thrml_spw.fashion_cnn_native.layer_allocation.common import OUT, sha, write, guard


def main():
    p = json.loads((OUT/'protocol.json').read_text())
    guard(p)
    assert not (OUT/'pipeline.json').exists()
    (OUT/'logs').mkdir(exist_ok=True)
    state = {'status': 'running', 'stages': []}
    for name, module, action in [('validation_production', 'study', 'run'), ('full_independent_validation_audit', 'verify', 'audit'), ('select_schedule', 'study', 'select'), ('independent_optimum', 'verify', 'selection')]:
        guard(p)
        command = [sys.executable, '-m', 'results.thrml_spw.fashion_cnn_native.layer_allocation.'+module, action]
        row = {'stage': name, 'command': command, 'status': 'running', 'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
        state['stages'].append(row)
        log = OUT/'logs'/(name+'.log')
        with log.open('wb') as stream:
            child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, env={**os.environ, 'OPENBLAS_NUM_THREADS': '4', 'OMP_NUM_THREADS': '4'})
            row['pid'] = child.pid
            write(OUT/'pipeline.json', state)
            code = child.wait()
        row.update(status='completed' if code == 0 else 'failed', returncode=code, finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), log_sha256=sha(log))
        write(OUT/'pipeline.json', state)
        if code:
            state['status'] = 'failed'
            write(OUT/'pipeline.json', state)
            raise SystemExit(code)
    state['status'] = 'validation_allocation_independently_closed_native_test_pending_goal_active'
    write(OUT/'pipeline.json', state)


if __name__ == '__main__':
    main()
