"""CPU training then full validation-only audit, separate from current GPU jobs."""
import datetime
import json
import os
import subprocess
import sys
from results.thrml_spw.cifar_cnn_native.train import OUT, sha, write, guard


def main():
    p = json.loads((OUT/'protocol.json').read_text())
    guard(p)
    assert not (OUT/'pipeline.json').exists()
    (OUT/'logs').mkdir(exist_ok=True)
    state = {'status': 'running', 'stages': []}
    for name, module, args in [('conventional_training', 'train', ['run']), ('independent_validation_audit', 'verify_training', [])]:
        guard(p)
        command = [sys.executable, '-m', 'results.thrml_spw.cifar_cnn_native.'+module]+args
        row = {'stage': name, 'command': command, 'status': 'running', 'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
        state['stages'].append(row)
        log = OUT/'logs'/(name+'.log')
        with log.open('wb') as stream:
            child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, env={**os.environ, 'OMP_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4'})
            row['pid'] = child.pid
            write(OUT/'pipeline.json', state)
            code = child.wait()
        row.update(status='completed' if code == 0 else 'failed', returncode=code, finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), log_sha256=sha(log))
        write(OUT/'pipeline.json', state)
        if code:
            state['status'] = 'failed'
            write(OUT/'pipeline.json', state)
            raise SystemExit(code)
    state['status'] = 'all5conventionalRGBcheckpoints_validation_audited_native_pending_goal_active'
    write(OUT/'pipeline.json', state)


if __name__ == '__main__':
    main()
