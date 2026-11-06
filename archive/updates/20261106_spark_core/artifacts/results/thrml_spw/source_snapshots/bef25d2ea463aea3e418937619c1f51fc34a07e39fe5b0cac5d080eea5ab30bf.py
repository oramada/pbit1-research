"""Seal the full RGB comparison, then wait for single-node audit closure."""
import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from results.thrml_spw.cifar_cnn_native.word_cohort.common import ROOT, TRAIN, OUT, PARENT, sha, write, guard, parent_closed


def parent_live():
    try:
        stat = Path('/proc/3964689/stat').read_text().rsplit(')', 1)[1].split()
        argv = Path('/proc/3964689/cmdline').read_bytes().split(b'\0')
        return stat[19] == '47733621' and stat[0] not in ['Z', 'X'] and b'results.thrml_spw.cifar_cnn_native.cohort.pipeline' in argv
    except FileNotFoundError:
        return False


def prepare():
    assert not (OUT/'protocol.json').exists()
    check = json.loads((OUT/'control_selfcheck.json').read_text())
    assert check['passed'] and len(check['rows']) == 5
    for path, expected in check['source_sha256'].items():
        assert sha(path) == expected, path
    witness = ROOT/'word_witness_evidence'
    assert json.loads((witness/'root_review.json').read_text())['passed']
    for path, expected in json.loads((witness/'review_completion.json').read_text())['artifact_sha256'].items():
        assert sha(path) == expected, path
    parent_status = json.loads((PARENT/'pipeline.json').read_text())['status']
    if parent_status == 'all150compactRGBnative_cases_independently_audited_goal_active':
        parent_closed()
    else:
        assert parent_status != 'failed' and parent_live()
    base = json.loads((witness/'protocol.json').read_text())
    sources = {**base['source_sha256'], **check['source_sha256']}
    for path in Path(__file__).parent.glob('*.py'):
        digest = sha(path)
        assert (Path('results/thrml_spw/source_snapshots')/(digest+'.py')).exists()
        sources[str(path)] = digest
    inputs = dict(base['input_sha256'])
    for path in [OUT/'control_selfcheck.json', witness/'review_completion.json', witness/'root_review.json',
                 witness/'completion.json', PARENT/'protocol.json', PARENT/'test_data.npz',
                 PARENT/'controls.json', PARENT/'control_logits.npz']:
        inputs[str(path)] = sha(path)
    p = {'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
         'source_sha256': sources, 'input_sha256': inputs, 'models': 5, 'MC_seeds': 10,
         'images': 10000, 'bank_size': 8, 'banks': 1250, 'chunk': 4, 'cases': 150,
         'word_samples': [16, 64, 256], 'single_samples': [64, 256, 1024],
         'K': 4, 'word_weight_nodes': 15712, 'single_weight_nodes': 3928,
         'key_root': 1777520, 'bootstrap_word_seed': 1787520, 'bootstrap_draws': 5000,
         'native_updates': 329952000000, 'public_updates': 329952000000,
         'counter_entries': 9237375000, 'word_warmup_updates_per_full_pass': 2946000000,
         'preflight_retained_updates_per_pass': 52792320, 'preflight_warmup_per_pass': 471360,
         'largest_gate_bound': 250675200, 'initial_minimum_free_bytes': 370000000,
         'case_minimum_free_bytes': 50000000, 'output_bound_bytes': 330000000,
         'contrasts': ['word_minus_single', 'word_minus_FP32', 'word_minus_INT4_minmax', 'word_minus_nominal_word_mean'],
         'statistics': 'All3budgets/all12contrasts, paired5model IDs using parent5000bootstrap model draws. Independent word MC index resampling from new1787520seed; parent single MC indices retained. Fixed historically consumed images, descriptive intervals only; no common random weight coupling, fresh generalization, equivalence or outcome-selected gate.',
         'representation': 'Fourfreepbits pernormalizedweight, nearest16levelcode with coefficients1/2/4/8 and finiteF32fields±4. Same per-output centers/radii/ACT8ranges/deterministicbiases as single. Integer output accumulation before externalF64readout/ReLU/maxpool. Internal comparator, not an asserted Extropic hardware weight format.',
         'scope': 'Full compact16x16RGB CIFAR model, moderate conventional baseline. Equal retained scheduled node evaluations, additional warmup/programming/controllers/state/readout costs separate. No physical systems advantage assumed. Parent full native/public/control/statistical closure precedes all word GPU/test execution.'}
    assert 15712*1250*50*sum(p['word_samples']) == p['native_updates']
    assert 49266*1250*150 == p['counter_entries']
    assert 255*256*15*256 == p['largest_gate_bound'] < 2**31
    guard(p)
    write(OUT/'protocol.json', p)
    print('Full150case RGB matched-resource protocol sealed before word test scoring', flush=True)


def run():
    p = json.loads((OUT/'protocol.json').read_text())
    guard(p)
    assert not (OUT/'pipeline.json').exists(), 'No duplicate queued comparison.'
    (OUT/'logs').mkdir()
    state = {'status': 'waiting_for_single_RGB_full_audits', 'stages': []}
    write(OUT/'pipeline.json', state)
    while True:
        guard(p)
        parent = json.loads((PARENT/'pipeline.json').read_text())
        if parent['status'] == 'all150compactRGBnative_cases_independently_audited_goal_active':
            parent_closed()
            break
        if parent['status'] == 'failed' or not parent_live():
            again = json.loads((PARENT/'pipeline.json').read_text())
            if again['status'] == 'all150compactRGBnative_cases_independently_audited_goal_active':
                parent_closed()
                break
            state.update(status='parent_terminal_before_audit_success', parent_status=again['status'])
            write(OUT/'pipeline.json', state)
            raise SystemExit(1)
        state['parent_live_observed_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        write(OUT/'pipeline.json', state)
        time.sleep(30)
    assert shutil.disk_usage('.').free >= p['initial_minimum_free_bytes']
    gpu = ['.venv/bin/python', '-m', 'research.thrml_spw.gpu_environment', '--', '.venv-thrml-gpu/bin/python', '-m']
    cpu, torch = ['.venv-thrml/bin/python', '-m'], ['.venv/bin/python', '-m']
    stages = [('GPU_validation', gpu, 'run', ['preflight']), ('nominal_controls', cpu, 'controls', ['run']),
              ('production', gpu, 'run', ['run']), ('public_replay', gpu, 'run', ['audit']),
              ('statistics', cpu, 'statistics', []), ('independent_statistics', torch, 'verify', [])]
    for name, runner, module, args in stages:
        guard(p)
        command = runner+['results.thrml_spw.cifar_cnn_native.word_cohort.'+module]+args
        row = {'stage': name, 'command': command, 'status': 'running', 'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
        state['status'] = 'running'
        state['stages'].append(row)
        log = OUT/'logs'/(name+'.log')
        with log.open('wb') as stream:
            child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                env={**os.environ, 'OPENBLAS_NUM_THREADS': '4', 'OMP_NUM_THREADS': '4', 'XLA_FLAGS': '--xla_gpu_autotune_level=0'})
            row['pid'] = child.pid
            write(OUT/'pipeline.json', state)
            code = child.wait()
        row.update(returncode=code, status='completed' if code == 0 else 'failed',
                   finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), log_sha256=sha(log))
        write(OUT/'pipeline.json', state)
        if code:
            state['status'] = 'failed'
            write(OUT/'pipeline.json', state)
            raise SystemExit(code)
    state['status'] = 'all150RGBword_cases_independently_audited_goal_active'
    write(OUT/'pipeline.json', state)


if __name__ == '__main__':
    {'prepare': prepare, 'run': run}[sys.argv[1]]()
