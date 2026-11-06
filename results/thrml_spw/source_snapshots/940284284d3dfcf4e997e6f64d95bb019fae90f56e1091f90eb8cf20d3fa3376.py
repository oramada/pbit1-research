"""Freeze once and wait for the actual matched-word parent before GPU work."""
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path('results/thrml_spw/fashion_cnn_native')
OUT = ROOT/'probability_native_evidence'
PARENT = ROOT/'word_comparison_evidence'
SUCCESS = 'full_matched_word_CNN_independently_audited_goal_active'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, obj):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2)+'\n')
    tmp.replace(path)


def guard(p):
    for mapping in [p['source_sha256'], p['input_sha256']]:
        for path, expected in mapping.items():
            assert sha(path) == expected, path


def seal(path):
    for name, expected in json.loads(Path(path).read_text())['artifact_sha256'].items():
        assert sha(name) == expected, name


def parent_closed():
    assert json.loads((PARENT/'pipeline.json').read_text())['status'] == SUCCESS
    seal(PARENT/'completion.json')
    assert json.loads((PARENT/'native_audit.json').read_text())['passed']
    assert json.loads((PARENT/'statistics_audit.json').read_text())['passed']


def freeze():
    OUT.mkdir(exist_ok=True)
    assert not (OUT/'protocol.json').exists()
    fine = ROOT/'probability_finetune_evidence'
    seal(fine/'preparation_completion.json')
    assert json.loads((fine/'evidence_audit.json').read_text())['passed']
    fine_manifest = json.loads((fine/'manifest.json').read_text())
    assert fine_manifest['complete'] and len(fine_manifest['rows']) == 10
    sources = json.loads((ROOT/'cohort_evidence/protocol.json').read_text())['source_sha256']
    sources.update(json.loads((fine/'protocol.json').read_text())['source_sha256'])
    sources.update({str(path): sha(path) for path in Path(__file__).parent.glob('*.py')})
    inputs = [fine/name for name in ['preparation_completion.json', 'protocol.json', 'manifest.json', 'evidence_audit.json']]
    inputs += [ROOT/'training/manifest.json', ROOT/'training/validation.npz', PARENT/'protocol.json']
    inputs += [Path(row['path']) for row in fine_manifest['rows']]
    inputs += [Path(row['path']) for row in json.loads((ROOT/'training/manifest.json').read_text())['rows']]
    parent_pid = 3800388
    proc = Path('/proc')/str(parent_pid)
    start = (proc/'stat').read_text().split()[21]
    assert start == '45280544' and b'word_comparison.pipeline' in (proc/'cmdline').read_bytes()
    p = {'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
         'source_sha256': sources, 'input_sha256': {str(path): sha(path) for path in inputs},
         'parent_pid': parent_pid, 'parent_start_ticks': start,
         'methods': ['expected_weight', 'binomial_mean_STE_T256'], 'models': 5,
         'samples': [64, 256, 1024], 'case_count': 30,
         'validation_images': 16, 'images_per_bank': 8, 'banks': 2,
         'key_root': 1647520, 'native_updates': 71608320, 'public_updates': 71608320,
         'counter_entries': 2264760, 'minimum_free_bytes': 50000000,
         'scope': 'OfficialTHRML validation-only prerequisite for all10probability-finetuned CNN exports, fixed originalc/r/bias and learneduint8fields. Public rawstates independently recover everyarray. No test scoring/quality selection/physical claims. Full native inference controls/statistics still required.'}
    guard(p)
    write(OUT/'protocol.json', p)


def run():
    p = json.loads((OUT/'protocol.json').read_text())
    guard(p)
    assert not (OUT/'pipeline.json').exists(), 'No implicit duplicate validation watcher'
    state = {'status': 'waiting_for_matched_word_full_audits', 'stages': []}
    write(OUT/'pipeline.json', state)
    while True:
        guard(p)
        parent = json.loads((PARENT/'pipeline.json').read_text())
        if parent['status'] == SUCCESS:
            parent_closed()
            break
        proc = Path('/proc')/str(p['parent_pid'])
        try:
            alive = (proc/'stat').read_text().split()[21] == p['parent_start_ticks'] and b'word_comparison.pipeline' in (proc/'cmdline').read_bytes()
        except (FileNotFoundError, ProcessLookupError):
            alive = False
        if parent['status'] == 'failed' or not alive:
            again = json.loads((PARENT/'pipeline.json').read_text())
            if again['status'] == SUCCESS:
                parent_closed()
                break
            state.update(status='parent_terminal_before_success', parent_status=again['status'], parent_live=alive)
            write(OUT/'pipeline.json', state)
            raise SystemExit(1)
        state['observed_parent_live_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        write(OUT/'pipeline.json', state)
        time.sleep(30)
    assert shutil.disk_usage('.').free >= p['minimum_free_bytes']
    command = ['.venv/bin/python', '-m', 'research.thrml_spw.gpu_environment', '--',
               '.venv-thrml-gpu/bin/python', '-m', 'results.thrml_spw.fashion_cnn_native.probability_native.preflight']
    row = {'stage': 'native_validation_and_public_recovery', 'command': command, 'status': 'running',
           'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
    state.update(status='running')
    state['stages'].append(row)
    with (OUT/'execution.log').open('wb') as stream:
        child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                 env={**os.environ, 'OPENBLAS_NUM_THREADS': '4', 'XLA_FLAGS': '--xla_gpu_autotune_level=0'})
        row['pid'] = child.pid
        write(OUT/'pipeline.json', state)
        code = child.wait()
    row.update(returncode=code, status='completed' if code == 0 else 'failed',
               log_sha256=sha(OUT/'execution.log'), finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    state['status'] = 'learned_probability_native_validation_audited_goal_active' if code == 0 else 'failed'
    write(OUT/'pipeline.json', state)
    raise SystemExit(code)


if __name__ == '__main__':
    {'freeze': freeze, 'run': run}[sys.argv[1]]()
