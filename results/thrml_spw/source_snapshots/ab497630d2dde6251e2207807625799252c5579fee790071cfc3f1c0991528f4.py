"""Fixed matched-update CNN test after validation-selected schedules are sealed."""
import datetime
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
import numpy as np
from results.thrml_spw.fashion_cnn_native.cohort.common import ROOT, TRAIN, sha, write, guard

COUNTS = np.array([72, 1152, 1440], np.int64)

OUT = ROOT/'allocated_native_evidence'
SELECTION = ROOT/'layer_allocation_evidence'
WAIT = ROOT/'probability_cohort_evidence'
BASE = ROOT/'cohort_evidence'


def seal(path):
    for file, expected in json.loads(path.read_text())['artifact_sha256'].items():
        assert sha(file) == expected, file


def prepare():
    OUT.mkdir(exist_ok=True)
    assert not (OUT/'protocol.json').exists()
    assert subprocess.check_output(['git', '-C', 'vendor/thrml', 'rev-parse', 'HEAD'], text=True).strip() == '2d54cd8404df85bb65703d7e8b4e84e48c1b72a9'
    assert not subprocess.check_output(['git', '-C', 'vendor/thrml', 'status', '--short'], text=True)
    seal(SELECTION/'completion.json')
    assert json.loads((SELECTION/'selection_audit.json').read_text())['passed']
    base = json.loads((BASE/'protocol.json').read_text())
    sources = dict(base['source_sha256'])
    sources.update({str(path): sha(path) for path in Path(__file__).parent.glob('*.py')})
    selected = json.loads((SELECTION/'allocation.json').read_text())
    schedules = [r['samples'] for r in selected['rows']]
    assert len(schedules) == 5 and all(int(COUNTS@row) == 681984 for row in schedules)
    inputs = dict(base['input_sha256'])
    paths = [SELECTION/n for n in ['protocol.json', 'completion.json', 'allocation.json', 'selection_audit.json']]
    paths += [BASE/n for n in ['completion.json', 'statistics.json', 'statistical_arrays.npz', 'test_data.npz']]
    paths += [WAIT/'protocol.json']
    inputs.update({str(path): sha(path) for path in paths})
    proc = Path('/proc/3851463')
    stat = proc.joinpath('stat').read_text().split(') ', 1)[1].split()
    assert stat[19] == '45624181' and b'probability_cohort.pipeline run' in proc.joinpath('cmdline').read_bytes().replace(b'\0', b' ')
    p = {'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
         'source_sha256': sources, 'input_sha256': inputs, 'schedules': schedules,
         'conditions': ['uniform256', 'validation_allocated'], 'models': 5, 'MC_seeds': 10,
         'images': 10000, 'banks': 1250, 'bank_size': 8, 'chunk': 4, 'case_count': 100,
         'key_root': 1707520, 'bootstrap_seed': 1717520, 'bootstrap_draws': 5000,
         'scheduled_updates_per_bank': 681984, 'native_updates': 85248000000, 'public_updates': 85248000000,
         'warmup_node_updates_per_full_pass': 333000000,
         'update_accounting': 'T-weighted counters count retained-sample scheduled free-node updates, not physical flips. SamplingSchedule also executes one warmup sweep perbank/layer;333000000additional warmup node updates perfull production/replay pass, equal acrossconditions. Preflight retained-sample and warmup work separately reported. No physical transition or energy equivalence claim.',
         'counter_entries': 4718250000, 'waiting_parent_pid': 3851463, 'waiting_parent_startticks': 45624181,
         'minimum_free_bytes': 190000000, 'all_future_outputs_bound_bytes': 175000000,
         'preflight': 'All5models with uniform64/256/1024 and frozen allocated schedules on first16validation images,2banks each. Native integer observer versus independent public raw states; all14canonical arrays byte-exact before test scoring. Preflight native and public updates are separately charged.',
         'statistics': 'All100cases on fixed consumed10000images,5paired original checkpoints/10freshMC seeds percondition, independent inference keys acrossconditions.5000hierarchical model/withincondition-seed draws, independent seed resampling acrossconditions; all3contrasts retained: allocated-minus-uniform and each-minus-FP32_F64. No image resampling, fresh generalization confirmation, equivalence or post-test schedule selection.',
         'scope': 'Native validation-selected sample redistribution from Gaussian validation sensitivity proxy. Exact equal weight-node update counts and same2664freeweightnodes, original8bitfields/ranges/ACT8/deterministicbiases/externalF64readout. Per-model3uint16schedule metadata6bytes/model must be counted. Max/serialized sample rounds may differ. Surrogate optimality does not imply native quality improvement or physical energy/latency/area benefit.'}
    guard(p)
    write(OUT/'protocol.json', p)


def backend():
    import jax
    from importlib.metadata import version
    assert version('thrml') == '0.1.4'
    assert subprocess.check_output(['git', '-C', 'vendor/thrml', 'rev-parse', 'HEAD'], text=True).strip() == '2d54cd8404df85bb65703d7e8b4e84e48c1b72a9'
    assert not subprocess.check_output(['git', '-C', 'vendor/thrml', 'status', '--short'], text=True)
    assert jax.default_backend() == 'gpu' and not jax.config.x64_enabled
    assert os.environ.get('XLA_FLAGS') == '--xla_gpu_autotune_level=0'


def preflight():
    from results.thrml_spw.fashion_cnn_native.inference.native import encode
    from results.thrml_spw.fashion_cnn_native.allocated_native.operator import forward, recover, key_bank
    from results.thrml_spw.fashion_cnn_native.compact.receipts import receipt
    backend()
    p = json.loads((OUT/'protocol.json').read_text())
    guard(p)
    assert not (OUT/'preflight.json').exists()
    x = np.load(TRAIN/'validation.npz')['inputs'][:16].astype(float).reshape(2, 8, 1, 14, 14)
    train = json.loads((TRAIN/'manifest.json').read_text())['rows']
    rows, updates, counter_entries = [], 0, 0
    (OUT/'preflight_cases').mkdir(exist_ok=True)
    for mi in range(5):
        model = np.load(train[mi]['path'])
        enc = encode(model)
        for ci, budgets in enumerate([[64]*3, [256]*3, [1024]*3, p['schedules'][mi]]):
            keys = [key_bank(0, mi, ci, 0, li, 0, 2) for li in range(3)]
            actual = forward(x, model, enc, budgets, keys)
            public = recover(x, model, budgets, keys)
            for name in actual:
                np.testing.assert_array_equal(actual[name], public[name])
            native_receipt, public_receipt = receipt(actual), receipt(public)
            np.testing.assert_array_equal(native_receipt, public_receipt)
            path = OUT/'preflight_cases'/f'model{mi}_condition{ci}.npz'
            with path.open('wb') as stream:
                np.savez_compressed(stream, **actual)
            work = 2*int(COUNTS@budgets)
            entries = sum(actual[f'gate_{li}'].size+actual[f'column_{li}'].size for li in range(3))
            updates += work
            counter_entries += entries
            rows.append({'model': mi, 'budgets': budgets, 'path': str(path), 'sha256': sha(path), 'native_updates': work, 'public_updates': work, 'counter_entries_exact': entries})
            print(json.dumps({'preflight_cases': len(rows)}), flush=True)
    guard(p)
    assert len(rows) == 20 and counter_entries == 1509840
    write(OUT/'preflight.json', {'passed': True, 'rows': rows, 'native_updates': updates, 'public_updates': updates,
                               'warmup_node_updates_per_pass': 106560,
                               'counter_entries_exact': counter_entries, 'protocol_sha256': sha(OUT/'protocol.json')})


def run(audit=False):
    from results.thrml_spw.fashion_cnn_native.inference.native import encode
    from results.thrml_spw.fashion_cnn_native.allocated_native.operator import forward, recover, key_bank
    from results.thrml_spw.fashion_cnn_native.compact.receipts import receipt
    backend()
    p = json.loads((OUT/'protocol.json').read_text())
    guard(p)
    assert json.loads((OUT/'preflight.json').read_text())['passed']
    assert shutil.disk_usage('.').free >= (50000000 if audit else p['minimum_free_bytes'])
    train = json.loads((TRAIN/'manifest.json').read_text())['rows']
    x = np.load(BASE/'test_data.npz')['inputs'].astype(float).reshape(1250, 8, 1, 14, 14)
    path = OUT/('native_audit.json' if audit else 'manifest.json')
    assert not path.exists()
    if audit:
        manifest = json.loads((OUT/'manifest.json').read_text())
        assert manifest['complete'] and len(manifest['rows']) == 100
        cases = manifest['rows']
    else:
        (OUT/'cases').mkdir(exist_ok=True)
        cases = [{'model': mi, 'condition': ci, 'seed': seed} for ci in range(2) for mi in range(5) for seed in range(10)]
    state = {'complete': False, 'rows': [], 'protocol_sha256': sha(OUT/'protocol.json'), 'scheduled_updates': 0, 'counter_entries': 0}
    progress = OUT/'audit_progress.json' if audit else path
    for row in cases:
        guard(p)
        mi, ci, seed = row['model'], row['condition'], row['seed']
        budgets = [256]*3 if ci == 0 else p['schedules'][mi]
        assert int(COUNTS@budgets) == p['scheduled_updates_per_bank']
        model = np.load(train[mi]['path'])
        enc = None if audit else encode(model)
        name = f'condition{ci}_model{mi}_seed{seed}'
        case = OUT/'cases'/(name+'.npz')
        if audit:
            assert sha(case) == row['sha256']
            retained = np.load(case)
        receipts, logits, clipped = np.empty((1250, 14, 32), np.uint8), np.empty((10000, 10)), np.zeros(3, np.int64)
        for first in range(0, 1250, 4):
            n = min(4, 1250-first)
            keys = [key_bank(1, mi, ci, seed, li, first, n) for li in range(3)]
            raw = recover(x[first:first+n], model, budgets, keys) if audit else forward(x[first:first+n], model, enc, budgets, keys)
            receipts[first:first+n] = receipt(raw)
            logits[first*8:(first+n)*8] = raw['output_2'].reshape(n*8, 10)
            state['counter_entries'] += sum(raw[f'gate_{li}'].size+raw[f'column_{li}'].size for li in range(3))
            for li, h in enumerate([x[first:first+n], raw['output_0'], raw['output_1']]):
                clipped[li] += np.count_nonzero((h < 0) | (h > float(model['activation_span'][li])))
        if audit:
            for key, value in [('receipts', receipts), ('logits', logits), ('clipped_coordinates', clipped)]:
                np.testing.assert_array_equal(retained[key], value)
        else:
            assert np.isfinite(logits).all()
            with case.open('wb') as stream:
                np.savez_compressed(stream, receipts=receipts, logits=logits, clipped_coordinates=clipped)
        state['scheduled_updates'] += 1250*int(COUNTS@budgets)
        state['rows'].append({'name': name, 'model': mi, 'condition': ci, 'seed': seed, 'budgets': budgets,
                              'path': str(case), 'sha256': sha(case), 'all_receipts_and_final_logits_exact': bool(audit)})
        write(progress, state)
        print(json.dumps({'audited' if audit else 'produced': len(state['rows'])}), flush=True)
    assert len(state['rows']) == 100 and state['scheduled_updates'] == 85248000000 and state['counter_entries'] == 4718250000
    state['complete'] = True
    if audit:
        state.update(passed=True, manifest_sha256=sha(OUT/'manifest.json'))
    guard(p)
    write(path, state)


if __name__ == '__main__':
    {'prepare': prepare, 'preflight': preflight, 'run': run, 'audit': lambda: run(True)}[sys.argv[1]]()
