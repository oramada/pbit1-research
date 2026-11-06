"""Freeze complete learned-probability cohort before scoring and wait on validation."""
import datetime
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from results.thrml_spw.fashion_cnn_native.probability_cohort.common import (
    ROOT, OUT, PARENT, BASE, WORD, FINE, METHODS, CONTRASTS, sha, write, guard, seal, parent_closed)

SUCCESS = 'learned_probability_native_validation_audited_goal_active'


def prepare():
    OUT.mkdir(exist_ok=True)
    assert not (OUT/'protocol.json').exists()
    for name in ['control_selfcheck.json', 'bootstrap_environment_check.json']:
        assert json.loads((OUT/name).read_text())['passed']
    assert json.loads((OUT/'control_selfcheck.json').read_text())['source_sha256'] == sha(Path(__file__).with_name('verify.py'))
    seal(FINE/'preparation_completion.json')
    parent = json.loads((PARENT/'protocol.json').read_text())
    sources = dict(parent['source_sha256'])
    sources.update({str(path): sha(path) for path in Path(__file__).parent.glob('*.py')})
    inputs = dict(parent['input_sha256'])
    inputs.update(json.loads((FINE/'preparation_completion.json').read_text())['artifact_sha256'])
    additional = [PARENT/'protocol.json', PARENT/'preparation_completion.json', BASE/'completion.json',
                  BASE/'statistics.json', BASE/'statistical_arrays.npz', BASE/'test_data.npz', WORD/'protocol.json',
                  OUT/'control_selfcheck.json', OUT/'bootstrap_environment_check.json',
                  OUT/'process_command_guard_preparation_failure.json']
    inputs.update({str(path): sha(path) for path in additional})
    pid = 3849492
    proc = Path('/proc')/str(pid)
    ticks = (proc/'stat').read_text().split()[21]
    assert ticks == '45594336' and b'probability_native.watch run' in (proc/'cmdline').read_bytes().replace(b'\0', b' ')
    assert subprocess.check_output(['git', '-C', 'vendor/thrml', 'rev-parse', 'HEAD'], text=True).strip() == '2d54cd8404df85bb65703d7e8b4e84e48c1b72a9'
    assert not subprocess.check_output(['git', '-C', 'vendor/thrml', 'status', '--short'], text=True)
    p = {'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'source_sha256': sources,
         'input_sha256': inputs, 'parent_pid': pid, 'parent_start_ticks': ticks,
         'methods': METHODS, 'models': 5, 'MC_seeds': 10, 'samples': [64, 256, 1024],
         'images': 10000, 'bank_size': 8, 'banks': 1250, 'chunk': 4, 'case_count': 300,
         'key_root': 1657520, 'bootstrap_seed': 1667520, 'bootstrap_draws': 5000, 'contrasts': CONTRASTS,
         'native_updates': 447552000000, 'public_updates': 447552000000, 'counter_entries': 14154750000,
         'minimum_free_before_controls': 480000000, 'minimum_free_before_native': 472000000,
         'minimum_free_before_case': 50000000,
         'storage': {'raw_case_arrays': 408000000, 'case_with_overhead_bound': 1370000,
                     'controls_statistics_support_bound': 19000000, 'all_outputs_bound': 430000000,
                     'receipt_semantics': 'Fourteen name/dtype/shape/data SHA256digests perbank plus retained finalF64logits; hashes require complete public raw-state regeneration and do not decode transient arrays.'},
         'statistics': 'All33contrasts, all300cases on fixed previously consumed10000images. Five starting-model IDs paired acrossmethods/reference. Ten independentMCseeds withinmethod/model; fresh method-specific bootstrapseedindices, original/word prior indices reused with common5000model draws. Independent resampling across inference methods, not shared rawbits. No image resampling, new generalization confirmation, outcome-selected method/budget or retroactive baselinegate pass.',
         'scope': 'Full officialTHRML fixed-range learned-command CNN confirmation after separateall10modelnativevalidation succeeds. Both predeclaredtrainingmethods/allmodels/budgets/MCseeds retained. Nominalmeancontrols, parentFP32/INT4/originalnative and internalpositionalword comparisons; not asserted Extropic architecture or physicalsystemsadvantage. TrainingF32surrogate differences and earlierfailedchecks remain retained.'}
    assert p['storage']['all_outputs_bound'] == 300*1370000+19000000
    guard(p)
    write(OUT/'protocol.json', p)
    print('Full300case learned-probability CNN protocol sealed before test scoring', flush=True)


def run():
    p = json.loads((OUT/'protocol.json').read_text())
    guard(p)
    assert not (OUT/'pipeline.json').exists(), 'No duplicate full learned-probability runner'
    OUT.joinpath('logs').mkdir(exist_ok=True)
    state = {'status': 'waiting_for_all_learned_probability_native_validation', 'stages': []}
    write(OUT/'pipeline.json', state)
    while True:
        guard(p)
        parent = json.loads((PARENT/'pipeline.json').read_text())
        if parent['status'] == SUCCESS:
            parent_closed()
            break
        proc = Path('/proc')/str(p['parent_pid'])
        try:
            alive = (proc/'stat').read_text().split()[21] == p['parent_start_ticks'] and b'probability_native.watch run' in (proc/'cmdline').read_bytes().replace(b'\0', b' ')
        except (FileNotFoundError, ProcessLookupError):
            alive = False
        if parent['status'] in ['failed', 'parent_terminal_before_success'] or not alive:
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
    assert shutil.disk_usage('.').free >= p['minimum_free_before_controls']
    cpu = ['.venv-thrml/bin/python', '-m']
    gpu = ['.venv/bin/python', '-m', 'research.thrml_spw.gpu_environment', '--', '.venv-thrml-gpu/bin/python', '-m']
    torch = ['.venv/bin/python', '-m']
    prefix = 'results.thrml_spw.fashion_cnn_native.probability_cohort.'
    stages = [('controls', cpu, 'run', ['controls']), ('production', gpu, 'run', ['production']),
              ('public_native_audit', gpu, 'run', ['audit']), ('statistics', cpu, 'run', ['statistics']),
              ('independent_controls_statistics_report', torch, 'verify', ['verify'])]
    for name, runner, module, args in stages:
        guard(p)
        command = runner+[prefix+module]+args
        row = {'stage': name, 'command': command, 'status': 'running',
               'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
        state['status'] = 'running'
        state['stages'].append(row)
        log = OUT/'logs'/(name+'.log')
        with log.open('wb') as stream:
            child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                     env={**os.environ, 'OPENBLAS_NUM_THREADS': '4', 'OMP_NUM_THREADS': '4',
                                          'XLA_FLAGS': '--xla_gpu_autotune_level=0'})
            row['pid'] = child.pid
            write(OUT/'pipeline.json', state)
            code = child.wait()
        row.update(returncode=code, status='completed' if code == 0 else 'failed', log_sha256=sha(log),
                   finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        write(OUT/'pipeline.json', state)
        if code:
            state['status'] = 'failed'
            write(OUT/'pipeline.json', state)
            raise SystemExit(code)
    state['status'] = 'full_probability_finetuned_CNN_independently_audited_goal_active'
    write(OUT/'pipeline.json', state)


if __name__ == '__main__':
    {'prepare': prepare, 'run': run}[sys.argv[1]]()
