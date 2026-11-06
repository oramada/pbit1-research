"""Seal all300learnedRGBcases before scoring; queue behind matched-word closure."""
import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from results.thrml_spw.cifar_cnn_native.probability_cohort.common import (
    ROOT, OUT, PARENT, BASE, WORD, FINE, METHODS, CONTRASTS, sha, write, guard, seal, parent_closed)

WORD_SUCCESS = 'all150RGBword_cases_independently_audited_goal_active'
SUCCESS = 'full_probability_finetuned_RGB_independently_audited_goal_active'


def word_live():
    try:
        stat = Path('/proc/3967754/stat').read_text().rsplit(')', 1)[1].split()
        argv = Path('/proc/3967754/cmdline').read_bytes().split(b'\0')
        return stat[19] == '47852613' and stat[0] not in ['Z', 'X'] and b'results.thrml_spw.cifar_cnn_native.word_cohort.pipeline' in argv
    except (FileNotFoundError, ProcessLookupError):
        return False


def prepare():
    assert not (OUT/'protocol.json').exists()
    for name in ['control_selfcheck.json', 'bootstrap_environment_check.json', 'statistics_selfcheck.json']:
        assert json.loads((OUT/name).read_text())['passed']
    assert json.loads((OUT/'control_selfcheck.json').read_text())['source_sha256'] == sha(Path(__file__).with_name('verify.py'))
    seal(FINE/'completion.json')
    seal(PARENT/'completion.json')
    seal(PARENT/'review_completion.json')
    seal(BASE/'completion.json')
    parent = json.loads((PARENT/'protocol.json').read_text())
    sources = dict(parent['source_sha256'])
    for path in Path(__file__).parent.glob('*.py'):
        digest = sha(path)
        assert (Path('results/thrml_spw/source_snapshots')/(digest+'.py')).exists()
        sources[str(path)] = digest
    inputs = dict(parent['input_sha256'])
    for path in [FINE/'completion.json', FINE/'preparation_root_review.json', PARENT/'completion.json',
                 PARENT/'review_completion.json', BASE/'completion.json', BASE/'statistics.json',
                 BASE/'statistical_arrays.npz', BASE/'test_data.npz', WORD/'protocol.json',
                 OUT/'control_selfcheck.json', OUT/'bootstrap_environment_check.json', OUT/'statistics_selfcheck.json',
                 ROOT/'probability_statistics_fixture_evidence/fixture_completion.json']:
        inputs[str(path)] = sha(path)
    status = json.loads((WORD/'pipeline.json').read_text())['status']
    if status == WORD_SUCCESS:
        parent_closed()
    else:
        assert status != 'failed' and word_live()
    p = {'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
         'source_sha256': sources, 'input_sha256': inputs, 'methods': METHODS,
         'models': 5, 'MC_seeds': 10, 'samples': [64, 256, 1024], 'images': 10000,
         'bank_size': 8, 'banks': 1250, 'chunk': 4, 'case_count': 300,
         'key_root': 1857520, 'GPU_validation_key_root': 1867520, 'bootstrap_seed': 1877520,
         'bootstrap_draws': 5000, 'contrasts': CONTRASTS,
         'native_updates': 659904000000, 'public_updates': 659904000000,
         'counter_entries': 18474750000, 'warmup_updates_per_full_pass': 1473000000,
         'GPU_preflight_updates_per_pass': 105584640, 'GPU_preflight_warmup_per_pass': 235680,
         'minimum_free_before_controls': 550000000, 'minimum_free_before_native': 500000000,
         'minimum_free_before_case': 50000000,
         'storage': {'raw_case_arrays': 408000000, 'case_with_overhead_bound': 1370000,
                     'controls_statistics_support_bound': 49000000, 'GPU_preflight_bound': 40000000,
                     'all_outputs_bound': 500000000},
         'statistics': 'All300cases/all33contrasts, paired5starting-model units and10inferenceMCseeds. Parent5000modelbootstrap draws reused; fresh independent method-specific seedindices. Original/word inference indices retained separately; no common rawbits. Fixed historically consumed10000images, no fresh generalization/equivalence/multiplicity-adjusted or outcome-selected success gate. Allmethods/budgets/negativeoutcomes retained.',
         'scope': 'Full learned-command RGB confirmation after frozen2.6936point quality-target miss; original miss is retained. Both probability-trainingmethods get8extraepochs, so comparisons against originalFP32/frozen/word have extra-training confounds; expected-method control shares parameterization/epochs/orders. Training uses biased surrogates, no native/unbiased-gradient claim. Moderate56%baseline/compact16x16RGBmodel, not competitive modernCIFARquality. Onefreepbit plus8bitcontrol/originalfixedranges/biases/ACT8/integeroutputcounters/externalF64readout. No physical energy/latency/area/controller-retention or totalmemory advantage inferred. Fullword audit precedes all new GPU/test stages.'}
    assert p['native_updates'] == 2*5*10*1250*3928*sum(p['samples'])
    assert p['counter_entries'] == 300*1250*49266
    assert p['storage']['all_outputs_bound'] == 300*1370000+49000000+40000000
    guard(p)
    write(OUT/'protocol.json', p)
    print('All300learnedRGBcases/33contrasts source-sealed before task scoring', flush=True)


def run():
    p = json.loads((OUT/'protocol.json').read_text())
    guard(p)
    assert not (OUT/'pipeline.json').exists()
    (OUT/'logs').mkdir()
    state = {'status': 'waiting_for_RGB_word_full_audits', 'stages': []}
    write(OUT/'pipeline.json', state)
    while True:
        guard(p)
        status = json.loads((WORD/'pipeline.json').read_text())['status']
        if status == WORD_SUCCESS:
            parent_closed()
            break
        if status == 'failed' or not word_live():
            again = json.loads((WORD/'pipeline.json').read_text())['status']
            if again == WORD_SUCCESS:
                parent_closed()
                break
            state.update(status='word_parent_terminal_before_success', parent_status=again)
            write(OUT/'pipeline.json', state)
            raise SystemExit(1)
        state['word_live_observed_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        write(OUT/'pipeline.json', state)
        time.sleep(30)
    assert shutil.disk_usage('.').free >= p['minimum_free_before_controls']
    gpu = ['.venv/bin/python', '-m', 'research.thrml_spw.gpu_environment', '--', '.venv-thrml-gpu/bin/python', '-m']
    cpu, torch = ['.venv-thrml/bin/python', '-m'], ['.venv/bin/python', '-m']
    stages = [('GPU_validation', gpu, 'run', ['GPU_preflight']), ('controls', cpu, 'run', ['controls']),
              ('production', gpu, 'run', ['production']), ('public_replay', gpu, 'run', ['audit']),
              ('statistics', cpu, 'run', ['statistics']), ('independent_controls_statistics', torch, 'verify', ['verify'])]
    for name, runner, module, args in stages:
        guard(p)
        command = runner+['results.thrml_spw.cifar_cnn_native.probability_cohort.'+module]+args
        row = {'stage': name, 'command': command, 'status': 'running',
               'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
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
    state['status'] = SUCCESS
    write(OUT/'pipeline.json', state)


if __name__ == '__main__':
    {'prepare': prepare, 'run': run}[sys.argv[1]]()
