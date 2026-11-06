"""Freeze and execute the complete T4096 RGB follow-up without automatic resume."""
import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from results.thrml_spw.cifar_cnn_native.probability_high_budget.common import (
    ROOT, OUT, PARENT, BASE, WORD, PRECISION, WITNESS, METHODS, CONTRASTS,
    sha, write, guard, seal, parent_closed)

SUCCESS = 'all100_T4096_RGB_cases_independently_audited_goal_active'


def gpu_jobs():
    found = []
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        try:
            argv = (path/'cmdline').read_bytes().split(b'\0')
            if b'.venv-thrml-gpu/bin/python' not in argv[0]:
                continue
            module = argv[argv.index(b'-m')+1]
            if module.startswith(b'results.thrml_spw.') or module.startswith(b'research.thrml_spw.'):
                state = (path/'stat').read_text().rsplit(')', 1)[1].split()
                if state[0] not in ['Z', 'X']:
                    found.append({'pid': int(path.name), 'start_ticks': state[19], 'module': module.decode()})
        except (FileNotFoundError, ProcessLookupError, PermissionError, IndexError, ValueError):
            continue
    return found


def prepare():
    assert not (OUT/'protocol.json').exists()
    parent_closed()
    check = json.loads((OUT/'statistics_selfcheck.json').read_text())
    assert check['passed'] and check['contrasts_checked'] == 15
    assert check['source_sha256'] == sha(Path(__file__).with_name('statistics.py'))
    previous = json.loads((PARENT/'protocol.json').read_text())
    sources, inputs = dict(previous['source_sha256']), dict(previous['input_sha256'])
    for folder in [PARENT, PRECISION, WITNESS]:
        parent = json.loads((folder/'protocol.json').read_text())
        sources.update(parent['source_sha256'])
        inputs.update(parent['input_sha256'])
        for name in ['completion.json', 'review_completion.json']:
            path = folder/name
            if not path.exists():
                continue
            seal(path)
            inputs[str(path)] = sha(path)
            inputs.update(json.loads(path.read_text())['artifact_sha256'])
        inputs[str(folder/'protocol.json')] = sha(folder/'protocol.json')
    for path in Path(__file__).parent.glob('*.py'):
        sources[str(path)] = sha(path)
        assert (Path('results/thrml_spw/source_snapshots')/(sha(path)+'.py')).exists()
    for path in [PARENT/'pipeline.json', PARENT/'statistical_arrays.npz', PARENT/'mean_logits.npz',
                 BASE/'statistical_arrays.npz', BASE/'test_data.npz', WORD/'statistical_arrays.npz',
                 PRECISION/'statistical_arrays.npz', PRECISION/'logits.npz',
                 WITNESS/'planned_task_followup.json', OUT/'statistics_selfcheck.json']:
        inputs[str(path)] = sha(path)
    p = {'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'source_sha256': sources, 'input_sha256': inputs, 'methods': METHODS,
        'models': 5, 'MC_seeds': 10, 'samples': [4096], 'images': 10000,
        'banks': 1250, 'bank_size': 8, 'chunk': 4, 'case_count': 100,
        'key_root': 1897520, 'bootstrap_seed': 1907520, 'bootstrap_draws': 5000,
        'contrasts': CONTRASTS, 'witness_protocol_sha256': sha(WITNESS/'protocol.json'),
        'native_updates': 2011136000000, 'public_updates': 2011136000000,
        'counter_entries': 6158250000, 'warmup_updates_per_full_pass': 491000000,
        'minimum_free_before_native': 250000000, 'minimum_free_before_case': 50000000,
        'storage': {'all_outputs_bound': 250000000, 'case_with_overhead_bound': 1370000,
                    'all100case_raw_arrays': 136000000, 'copied_controls_statistics_logs_bound': 113000000},
        'statistics': 'All100cases/all15contrasts retained. Five paired starting-model units and ten inference seeds,5000inherited modelbootstrap draws, fresh method-specific MCbootstrap indices. OriginalT1024/wordT256 seedindices retained independently. Fixed historically consumed images; no fresh-data confirmation, formal equivalence/noninferiority margin/test, multiplicity adjustment, outcome-selected success gate or independent new training replicas. Samples4096 selected after previousT1024 outcomes to investigate residual sampling error.',
        'scope': 'Onefreeweightpbit per normalized parameter;3928weightnodes/34deterministicbiases/originalfixedranges/unchangedlearned8bitcommands/ACT8spans/int32counterobserver/externalF64readout. Both training methods retained, each has8extraepochs versus frozenFP32/BF16/original/word references. Expected-weight method shares parameterization/epochs/orders with sampled surrogate. Training remains biased surrogate, not native or unbiased. This extends sampling4x versusT1024 without altering learnedfits/controls. Same moderatecompact16x16RGBbaseline. Original failures/lowerbudgetresults remain unchanged. Equalnode/area/physicalenergy/latency/controllerprecision advantage not established.'}
    assert p['native_updates'] == 100*1250*3928*4096
    assert p['counter_entries'] == 100*1250*49266
    assert p['warmup_updates_per_full_pass'] == 100*1250*3928
    assert len(p['contrasts']) == 15
    guard(p)
    assert not gpu_jobs(), 'Do not overlap another active scientific GPU worker'
    assert shutil.disk_usage('.').free >= p['storage']['all_outputs_bound']
    write(OUT/'protocol.json', p)
    print('All100T4096cases/15comparisons frozen before new test scoring')


def run():
    p = json.loads((OUT/'protocol.json').read_text())
    guard(p)
    parent_closed()
    assert not (OUT/'pipeline.json').exists() and not gpu_jobs()
    (OUT/'logs').mkdir()
    state = {'status': 'running', 'stages': []}
    write(OUT/'pipeline.json', state)
    gpu = ['.venv/bin/python', '-m', 'research.thrml_spw.gpu_environment', '--', '.venv-thrml-gpu/bin/python', '-m']
    cpu = ['.venv/bin/python', '-m']
    stages = [('controls_copy', cpu, 'run', ['controls']), ('production', gpu, 'run', ['production']),
              ('public_replay', gpu, 'run', ['audit']), ('statistics', cpu, 'statistics', ['run']),
              ('independent_statistics', cpu, 'verify', [])]
    for name, runner, module, args in stages:
        guard(p)
        assert shutil.disk_usage('.').free >= p['minimum_free_before_native']
        command = runner+['results.thrml_spw.cifar_cnn_native.probability_high_budget.'+module]+args
        row = {'stage': name, 'status': 'running', 'command': command,
               'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
        state['stages'].append(row)
        log = OUT/'logs'/(name+'.log')
        with log.open('wb') as stream:
            child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                env={**os.environ, 'OPENBLAS_NUM_THREADS': '4', 'OMP_NUM_THREADS': '4',
                     'XLA_FLAGS': '--xla_gpu_autotune_level=0'})
            row['pid'] = child.pid
            row['start_ticks'] = Path(f'/proc/{child.pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
            write(OUT/'pipeline.json', state)
            code = child.wait()
        row.update(returncode=code, status='completed' if code == 0 else 'failed',
                   finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), log_sha256=sha(log))
        write(OUT/'pipeline.json', state)
        if code:
            state['status'] = 'failed'
            write(OUT/'pipeline.json', state)
            raise SystemExit(code)
    guard(p)
    seal(OUT/'completion.json')
    state['status'] = SUCCESS
    write(OUT/'pipeline.json', state)


if __name__ == '__main__':
    {'prepare': prepare, 'run': run}[sys.argv[1]]()
