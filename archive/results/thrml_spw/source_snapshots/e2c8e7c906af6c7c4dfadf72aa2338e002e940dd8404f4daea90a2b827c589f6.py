"""Wait on actual native-allocation process and then close all fixed statistics."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from results.thrml_spw.fashion_cnn_native.cohort.common import sha, write, guard
from results.thrml_spw.fashion_cnn_native.allocated_statistics.study import OUT, PARENT


def main():
    p = json.loads((OUT/'protocol.json').read_text())
    guard(p)
    assert json.loads((OUT/'selfcheck.json').read_text())['passed']
    assert not (OUT/'pipeline.json').exists()
    (OUT/'logs').mkdir(exist_ok=True)
    state = {'status': 'waiting_for_live_native_allocation_pipeline', 'stages': []}
    write(OUT/'pipeline.json', state)
    success = 'all100native_allocated_comparisons_publicly_replayed_statistics_pending_goal_active'
    while True:
        guard(p)
        parent = json.loads((PARENT/'pipeline.json').read_text())
        if parent['status'] == success:
            assert json.loads((PARENT/'native_audit.json').read_text())['passed']
            break
        proc = Path('/proc')/str(p['parent_pipeline_pid'])
        try:
            stat = proc.joinpath('stat').read_text().split(') ', 1)[1].split()
            live = int(stat[19]) == p['parent_pipeline_startticks'] and stat[0] != 'Z' and b'allocated_native.pipeline' in proc.joinpath('cmdline').read_bytes()
        except (FileNotFoundError, ProcessLookupError):
            live = False
        if parent['status'] in ['failed', 'prior_pipeline_terminal_without_success'] or not live:
            again = json.loads((PARENT/'pipeline.json').read_text())
            if again['status'] == success:
                break
            state.update(status='native_parent_terminal_without_success', parent_status=again['status'], parent_live=live)
            write(OUT/'pipeline.json', state)
            raise SystemExit(1)
        state['parent_observed_live_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        write(OUT/'pipeline.json', state)
        time.sleep(30)
    for name, module, action in [('statistics', 'study', 'run'), ('independent_verification', 'verify', 'verify')]:
        guard(p)
        command = [sys.executable, '-m', 'results.thrml_spw.fashion_cnn_native.allocated_statistics.'+module, action]
        row = {'stage': name, 'command': command, 'status': 'running', 'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
        state.update(status='running')
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
    state['status'] = 'all100matched_budget_CNN_cases_statistically_closed_goal_active'
    write(OUT/'pipeline.json', state)


if __name__ == '__main__':
    main()
