"""Freeze and execute all400same-fit word cases, with no automatic resume."""
import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from results.thrml_spw.cifar_cnn_native.learned_word_cohort.common import (
    ROOT, OUT, LOW, HIGH, SAME, BASE, WITNESS, CONTRASTS, sha, write, guard, seal, closed, gpu_jobs)

SUCCESS = 'all400_samefit_word_cases_independently_verified_goal_active'


def prepare():
    assert not (OUT/'protocol.json').exists(); closed()
    control_check = json.loads((OUT/'controls_selfcheck.json').read_text())
    stat_check = json.loads((OUT/'statistics_selfcheck.json').read_text())
    assert control_check['passed'] and len(control_check['rows']) == 10 and stat_check['passed']
    assert control_check['source_sha256'] == sha(Path(__file__).with_name('verify.py'))
    assert stat_check['source_sha256'] == sha(Path(__file__).with_name('statistics.py'))
    sources = {}; inputs = {}
    for folder in [LOW, HIGH, SAME, WITNESS]:
        parent = json.loads((folder/'protocol.json').read_text())
        sources.update(parent['source_sha256']); inputs.update(parent['input_sha256'])
        for name in ['completion.json', 'review_completion.json']:
            inputs[str(folder/name)] = sha(folder/name)
            inputs.update(json.loads((folder/name).read_text())['artifact_sha256'])
    extras = [Path('results/thrml_spw/fashion_cnn_native/word_comparison/verify.py'),
              Path('results/thrml_spw/cifar_cnn_native/learned_word_v2/adapter.py'),
              Path('results/thrml_spw/cifar_cnn_native/learned_word_v2/replay.py')]
    for path in [*Path(__file__).parent.glob('*.py'), *extras]:
        sources[str(path)] = sha(path)
        assert (ROOT.parent/'source_snapshots'/(sha(path)+'.py')).exists()
    for path in [LOW/'statistical_arrays.npz', HIGH/'statistical_arrays.npz', SAME/'statistical_arrays.npz',
                 BASE/'statistical_arrays.npz', BASE/'test_data.npz', OUT/'controls_selfcheck.json', OUT/'statistics_selfcheck.json']:
        inputs[str(path)] = sha(path)
    p = {'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'source_sha256': sources, 'input_sha256': inputs, 'word_samples': [16, 64, 256, 1024],
        'matched_single_samples': [64, 256, 1024, 4096], 'methods': 2, 'models': 5,
        'inference_seeds': 10, 'test_images': 10000, 'banks': 1250, 'bank_size': 8,
        'cases': 400, 'key_root': 1927520, 'bootstrap_word_seed': 1937520, 'bootstrap_draws': 5000,
        'contrasts': CONTRASTS, 'contrast_rows': 24,
        'retained_updates_per_pass': 2671040000000, 'warmup_updates_per_pass': 7856000000,
        'counter_entries': 24633000000, 'minimum_free_before_stage': 800000000,
        'minimum_free_before_case': 100000000,
        'output_bounds_bytes': {'cases': 600000000, 'support': 200000000, 'total': 800000000},
        'statistics': 'All400cases/24contrasts retained; five paired starting-model units and ten MC inference seeds,5000inherited paired modelbootstrap draws. Independent fresh word MCbootstrap indices per method and original corresponding single MCbootstrap indices reused; no false pairing of independent random streams. Test images fixed and historically consumed. Retrospective baseline extension, no new training, favorable-method selection, formal equivalence margin/test, multiplicity adjustment or fresh-data confirmation.',
        'scope': 'Compare same ten learned fits with original centers/radii/deterministicbiases/ACT8spans. One free node per normalized weight at T64/256/1024/4096 versus four free positional nodes at T16/64/256/1024, finite fields±4,coefficients1/2/4/8,denominator15. Retained evaluations equal, warmup separately charged. Native output-counter observer before externalF64readout/nonlinearities; public raw-state causal replay. Internal reference, not asserted Extropic architecture. Command/range/bias/activation/control/counter/readout resources remain costs; no measured energy/latency/area/totalmemory advantage.'}
    assert 2*5*10*1250*15712*sum(p['word_samples']) == p['retained_updates_per_pass']
    assert p['cases']*1250*15712 == p['warmup_updates_per_pass']
    assert p['cases']*61582500 == p['counter_entries']
    assert 400*1360024 < p['output_bounds_bytes']['cases']
    guard(p); assert not gpu_jobs() and shutil.disk_usage('.').free >= p['output_bounds_bytes']['total']
    write(OUT/'protocol.json', p)
    print('All400same-fit word cases and24matched-work contrasts frozen before new task scoring')


def run():
    p = json.loads((OUT/'protocol.json').read_text()); guard(p); closed()
    assert not (OUT/'pipeline.json').exists() and not gpu_jobs()
    (OUT/'logs').mkdir(); state = {'status': 'running', 'stages': []}; write(OUT/'pipeline.json', state)
    cpu = ['.venv/bin/python', '-m']
    gpu = ['.venv/bin/python', '-m', 'research.thrml_spw.gpu_environment', '--', '.venv-thrml-gpu/bin/python', '-m']
    stages = [('controls', cpu, 'run', ['controls']), ('production', gpu, 'run', ['production']),
        ('public_replay', gpu, 'run', ['audit']), ('statistics', cpu, 'statistics', ['run']),
        ('independent_statistics', cpu, 'verify', ['run'])]
    for name, runner, module, args in stages:
        guard(p); assert shutil.disk_usage('.').free >= p['minimum_free_before_stage']
        command = runner+['results.thrml_spw.cifar_cnn_native.learned_word_cohort.'+module]+args
        row = {'stage': name, 'status': 'running', 'command': command,
            'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
        state['stages'].append(row); log = OUT/'logs'/(name+'.log')
        with log.open('wb') as stream:
            child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                env={**os.environ, 'OPENBLAS_NUM_THREADS': '4', 'OMP_NUM_THREADS': '4', 'XLA_FLAGS': '--xla_gpu_autotune_level=0'})
            row['pid'] = child.pid
            row['start_ticks'] = Path(f'/proc/{child.pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
            write(OUT/'pipeline.json', state); code = child.wait()
        row.update(returncode=code, status='completed' if code == 0 else 'failed',
            finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), log_sha256=sha(log))
        write(OUT/'pipeline.json', state)
        if code:
            state['status'] = 'failed'; write(OUT/'pipeline.json', state); raise SystemExit(code)
    guard(p); seal(OUT/'completion.json'); state['status'] = SUCCESS; write(OUT/'pipeline.json', state)


if __name__ == '__main__':
    {'prepare': prepare, 'run': run}[sys.argv[1]]()
