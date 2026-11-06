"""Wait for the actual fine-tuned pipeline before using the GPU, then run fixed study."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from results.thrml_spw.fashion_cnn_native.allocated_native.run import OUT, WAIT, SELECTION, seal
from results.thrml_spw.fashion_cnn_native.cohort.common import sha, write, guard


def main():
    p = json.loads((OUT/'protocol.json').read_text())
    guard(p)
    assert not (OUT/'pipeline.json').exists()
    (OUT/'logs').mkdir(exist_ok=True)
    state = {'status': 'waiting_for_live_finetuned_pipeline', 'stages': []}
    write(OUT/'pipeline.json', state)
    while True:
        guard(p)
        prior = json.loads((WAIT/'pipeline.json').read_text())
        if prior['status'] == 'full_probability_finetuned_CNN_independently_audited_goal_active':
            seal(WAIT/'completion.json')
            seal(SELECTION/'completion.json')
            break
        proc = Path('/proc')/str(p['waiting_parent_pid'])
        try:
            stat = proc.joinpath('stat').read_text().split(') ', 1)[1].split()
            live = int(stat[19]) == p['waiting_parent_startticks'] and stat[0] != 'Z' and b'probability_cohort.pipeline run' in proc.joinpath('cmdline').read_bytes().replace(b'\0', b' ')
        except (FileNotFoundError, ProcessLookupError):
            live = False
        if prior['status'] in ['failed', 'parent_terminal_before_success'] or not live:
            again = json.loads((WAIT/'pipeline.json').read_text())
            if again['status'] == 'full_probability_finetuned_CNN_independently_audited_goal_active':
                seal(WAIT/'completion.json')
                seal(SELECTION/'completion.json')
                break
            state.update(status='prior_pipeline_terminal_without_success', observed_parent_status=again['status'], parent_live=live)
            write(OUT/'pipeline.json', state)
            raise SystemExit(1)
        state['parent_observed_live_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        write(OUT/'pipeline.json', state)
        time.sleep(30)
    runner = ['.venv/bin/python', '-m', 'research.thrml_spw.gpu_environment', '--', '.venv-thrml-gpu/bin/python', '-m', 'results.thrml_spw.fashion_cnn_native.allocated_native.run']
    for stage in ['preflight', 'run', 'audit']:
        guard(p)
        command = runner+[stage]
        row = {'stage': stage, 'command': command, 'status': 'running', 'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
        state.update(status='running')
        state['stages'].append(row)
        log = OUT/'logs'/(stage+'.log')
        with log.open('wb') as stream:
            child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, env={**os.environ, 'OPENBLAS_NUM_THREADS': '4', 'OMP_NUM_THREADS': '4', 'XLA_FLAGS': '--xla_gpu_autotune_level=0'})
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
    state['status'] = 'all100native_allocated_comparisons_publicly_replayed_statistics_pending_goal_active'
    write(OUT/'pipeline.json', state)


if __name__ == '__main__':
    main()
