"""Sequential full-budget GPU witness, controls, native cases, replay and statistics."""
import datetime
import json
import os
import shutil
import subprocess
from results.thrml_spw.cifar_cnn_native.cohort.common import OUT, sha, write, guard, preparation_closed


def main():
    p = json.loads((OUT/'protocol.json').read_text())
    guard(p)
    preparation_closed()
    assert not (OUT/'pipeline.json').exists()
    assert shutil.disk_usage('.').free >= p['minimum_free_bytes']
    (OUT/'logs').mkdir(exist_ok=True)
    state = {'status': 'running', 'stages': []}
    gpu = ['.venv/bin/python', '-m', 'research.thrml_spw.gpu_environment', '--', '.venv-thrml-gpu/bin/python', '-m']
    cpu, torch = ['.venv-thrml/bin/python', '-m'], ['.venv/bin/python', '-m']
    stages = [('full_budget_GPU_preflight', gpu, 'run', ['preflight']), ('controls', torch, 'controls', ['run']),
              ('production', gpu, 'run', ['run']), ('public_native_replay', gpu, 'run', ['audit']),
              ('statistics', cpu, 'statistics', []), ('independent_controls_statistics', cpu, 'verify', [])]
    for name, runner, module, args in stages:
        guard(p)
        command = runner+['results.thrml_spw.cifar_cnn_native.cohort.'+module]+args
        row = {'stage': name, 'command': command, 'status': 'running', 'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
        state['stages'].append(row)
        log = OUT/'logs'/(name+'.log')
        with log.open('wb') as stream:
            child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                     env={**os.environ, 'OPENBLAS_NUM_THREADS': '4', 'OMP_NUM_THREADS': '4', 'XLA_FLAGS': '--xla_gpu_autotune_level=0'})
            row['pid'] = child.pid
            write(OUT/'pipeline.json', state)
            code = child.wait()
        row.update(status='completed' if code == 0 else 'failed', returncode=code,
                   finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), log_sha256=sha(log))
        write(OUT/'pipeline.json', state)
        if code:
            state['status'] = 'failed'
            write(OUT/'pipeline.json', state)
            raise SystemExit(code)
    state['status'] = 'all150compactRGBnative_cases_independently_audited_goal_active'
    write(OUT/'pipeline.json', state)


if __name__ == '__main__':
    main()
