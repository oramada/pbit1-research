"""Full RGB native production/public replay, gated by full-budget validation."""
import datetime
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
from results.thrml_spw.cifar_cnn_native.cohort.common import ROOT, TRAIN, OUT, CONTROLS, sha, write, save, guard, key_bank, preparation_closed


def prepare():
    OUT.mkdir(exist_ok=True)
    assert not (OUT/'protocol.json').exists()
    preparation_closed()
    check = json.loads((OUT/'control_selfcheck.json').read_text())
    assert check['passed'] and len(check['rows']) == 50
    for path, expected in check['source_sha256'].items():
        assert sha(path) == expected
    parent = json.loads((ROOT/'native_witness_evidence/protocol.json').read_text())
    sources = dict(parent['source_sha256'])
    sources.update(json.loads((TRAIN/'protocol.json').read_text())['source_sha256'])
    sources.update({str(path): sha(path) for path in Path(__file__).parent.glob('*.py')})
    inputs = dict(json.loads((TRAIN/'protocol.json').read_text())['input_sha256'])
    inputs.update(json.loads((ROOT/'preparation_completion.json').read_text())['artifact_sha256'])
    paths = [ROOT/'preparation_completion.json', TRAIN/'manifest.json', TRAIN/'validation.npz',
             ROOT/'native_witness_evidence/completion.json', OUT/'control_selfcheck.json',
             Path('data/torchvision/cifar-10-batches-py/test_batch')]
    inputs.update({str(path): sha(path) for path in paths})
    for row in json.loads((TRAIN/'manifest.json').read_text())['rows']:
        inputs[row['path']] = row['sha256']
    p = {'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'source_sha256': sources,
         'input_sha256': inputs, 'models': 5, 'MC_seeds': 10, 'samples': [64, 256, 1024],
         'images': 10000, 'bank_size': 8, 'banks': 1250, 'chunk': 4, 'case_count': 150,
         'key_root': 1737520, 'bootstrap_seed': 1747520, 'bootstrap_draws': 5000,
         'weight_nodes': 3928, 'deterministic_biases': 34, 'controls': CONTROLS,
         'native_updates': 329952000000, 'public_updates': 329952000000,
         'warmup_node_updates_per_full_pass': 736500000, 'counter_entries': 9237375000,
         'counter_entries_per_bank': 49266, 'largest_gate_bound': 66846720,
         'preflight_native_updates': 52792320, 'preflight_public_updates': 52792320,
         'preflight_warmup_updates_per_pass': 117840,
         'minimum_free_bytes': 370000000, 'case_minimum_free_bytes': 50000000,
         'output_bound_bytes': 330000000, 'raw_case_receipts_and_logits_bytes': 204000000,
         'preflight': 'All5selectedRGB models/first16validation inputs/T64,256,1024;15cases/2banks. All14native arrays must match public recovery byte-exact before any test baseline/task execution.',
         'representation': 'All3928signedweights use onefreeprobabilitypbit each, per-output center/radius and nearest8bitfields±4. Biases deterministic; ACT8 and exactpadding masks; integer temporal output counters before externalF64signed readout/ReLU/maxpool. No operational per-weight reconstruction.',
         'statistics': 'All150cases/all30control contrasts,5model units/10MCseeds each,5000hierarchical model/seed bootstrap draws with common percase model/seedindices across contrasts. Fixed historically consumed10000test images, no image resampling/fresh generalization confirmation/equivalence or favorable method/budget selection.',
         'scope': 'Full compact16x16RGB CIFAR10CNN, different smaller model than old softwarepilot. Moderate validation baseline~55–57% is explicit; not modern competitive CIFARquality or hardware benefit. All outcomes retained even if quality degrades. Warmup/production/public/preflight separately charged, scheduled updates are not physical flips.',
         'control_tolerances': {'F64_rtol': 2e-12, 'F64_atol': 2e-11, 'F32_rtol': 2e-5, 'F32_atol': 1e-4}}
    assert p['native_updates'] == 3928*1250*sum(p['samples'])*50
    assert p['counter_entries'] == 49266*1250*150
    assert subprocess.check_output(['git', '-C', 'vendor/thrml', 'rev-parse', 'HEAD'], text=True).strip() == '2d54cd8404df85bb65703d7e8b4e84e48c1b72a9'
    assert not subprocess.check_output(['git', '-C', 'vendor/thrml', 'status', '--short'], text=True)
    guard(p)
    write(OUT/'protocol.json', p)


def backend():
    import jax
    assert version('thrml') == '0.1.4' and jax.default_backend() == 'gpu' and not jax.config.x64_enabled
    assert os.environ.get('XLA_FLAGS') == '--xla_gpu_autotune_level=0'
    assert subprocess.check_output(['git', '-C', 'vendor/thrml', 'rev-parse', 'HEAD'], text=True).strip() == '2d54cd8404df85bb65703d7e8b4e84e48c1b72a9'
    assert not subprocess.check_output(['git', '-C', 'vendor/thrml', 'status', '--short'], text=True)


def preflight():
    from results.thrml_spw.fashion_cnn_native.inference.native import encode
    from results.thrml_spw.fashion_cnn_native.allocated_native.operator import forward, recover
    from results.thrml_spw.fashion_cnn_native.compact.receipts import receipt
    backend()
    p = json.loads((OUT/'protocol.json').read_text())
    guard(p)
    assert not (OUT/'preflight.json').exists()
    x = np.load(TRAIN/'validation.npz')['inputs'][:16].astype(float).reshape(2, 8, 3, 16, 16)
    rows, updates, entries = [], 0, 0
    (OUT/'preflight_cases').mkdir(exist_ok=True)
    for mi, row in enumerate(json.loads((TRAIN/'manifest.json').read_text())['rows']):
        model = np.load(row['path'])
        enc = encode(model)
        for samples in p['samples']:
            keys = [key_bank(0, mi, 0, samples, li, 0, 2) for li in range(3)]
            native, public = forward(x, model, enc, [samples]*3, keys), recover(x, model, [samples]*3, keys)
            for name in native:
                np.testing.assert_array_equal(native[name], public[name])
            np.testing.assert_array_equal(receipt(native), receipt(public))
            path = OUT/'preflight_cases'/f'model{mi}_T{samples}.npz'
            save(path, **native)
            updates += 2*3928*samples
            count = sum(native[f'gate_{li}'].size+native[f'column_{li}'].size for li in range(3))
            entries += count
            rows.append({'model': mi, 'T': samples, 'path': str(path), 'sha256': sha(path), 'counter_entries_exact': count})
            print(json.dumps({'GPU_RGB_preflight_cases': len(rows)}), flush=True)
    assert len(rows) == 15 and updates == 52792320 and entries == 1477980
    guard(p)
    write(OUT/'preflight.json', {'passed': True, 'rows': rows, 'native_updates': updates, 'public_updates': updates,
                               'counter_entries_exact': entries, 'protocol_sha256': sha(OUT/'protocol.json')})


def run(audit=False):
    from results.thrml_spw.fashion_cnn_native.inference.native import encode
    from results.thrml_spw.fashion_cnn_native.allocated_native.operator import forward, recover
    from results.thrml_spw.fashion_cnn_native.compact.receipts import receipt
    backend()
    p = json.loads((OUT/'protocol.json').read_text())
    guard(p)
    assert json.loads((OUT/'preflight.json').read_text())['passed']
    controls = json.loads((OUT/'controls.json').read_text())
    assert controls['complete'] and controls['test_data_sha256'] == sha(OUT/'test_data.npz')
    destination = OUT/('native_audit.json' if audit else 'manifest.json')
    assert not destination.exists()
    assert shutil.disk_usage('.').free >= p['case_minimum_free_bytes']
    x = np.load(OUT/'test_data.npz')['inputs'].astype(float).reshape(1250, 8, 3, 16, 16)
    train = json.loads((TRAIN/'manifest.json').read_text())['rows']
    if audit:
        m = json.loads((OUT/'manifest.json').read_text())
        assert m['complete'] and len(m['rows']) == 150
        cases = m['rows']
    else:
        (OUT/'cases').mkdir(exist_ok=True)
        cases = [{'model': mi, 'T': t, 'seed': seed} for mi in range(5) for t in p['samples'] for seed in range(10)]
    state = {'complete': False, 'protocol_sha256': sha(OUT/'protocol.json'), 'rows': [],
             'scheduled_updates': 0, 'counter_entries': 0, 'test_data_sha256': sha(OUT/'test_data.npz')}
    progress = OUT/'audit_progress.json' if audit else destination
    for row in cases:
        guard(p)
        assert shutil.disk_usage('.').free >= p['case_minimum_free_bytes']
        mi, samples, seed = row['model'], row['T'], row['seed']
        model = np.load(train[mi]['path'])
        enc = None if audit else encode(model)
        name = f'model{mi}_T{samples}_seed{seed}'
        path = OUT/'cases'/(name+'.npz')
        if audit:
            assert sha(path) == row['sha256']
            case = np.load(path)
        hashes, logits, clipped = np.empty((1250, 14, 32), np.uint8), np.empty((10000, 10)), np.zeros(3, np.int64)
        for first in range(0, 1250, 4):
            count = min(4, 1250-first)
            keys = [key_bank(1, mi, seed, samples, li, first, count) for li in range(3)]
            native = recover(x[first:first+count], model, [samples]*3, keys) if audit else forward(x[first:first+count], model, enc, [samples]*3, keys)
            hashes[first:first+count] = receipt(native)
            logits[first*8:(first+count)*8] = native['output_2'].reshape(count*8, 10)
            state['counter_entries'] += sum(native[f'gate_{li}'].size+native[f'column_{li}'].size for li in range(3))
            for li, h in enumerate([x[first:first+count], native['output_0'], native['output_1']]):
                clipped[li] += np.count_nonzero((h < 0) | (h > float(model['activation_span'][li])))
        if audit:
            for key, value in [('receipts', hashes), ('logits', logits), ('clipped_coordinates', clipped)]:
                np.testing.assert_array_equal(value, case[key])
        else:
            assert np.isfinite(logits).all()
            save(path, receipts=hashes, logits=logits, clipped_coordinates=clipped)
        state['scheduled_updates'] += 1250*3928*samples
        state['rows'].append({'name': name, 'model': mi, 'T': samples, 'seed': seed, 'path': str(path),
                              'sha256': sha(path), 'clipped_coordinates': clipped.tolist(),
                              'all_receipts_and_final_logits_exact': bool(audit)})
        write(progress, state)
        print(json.dumps({'audited' if audit else 'produced': len(state['rows']), 'last': name}), flush=True)
    assert len(state['rows']) == 150 and state['scheduled_updates'] == 329952000000 and state['counter_entries'] == 9237375000
    state['complete'] = True
    if audit:
        state.update(passed=True, manifest_sha256=sha(OUT/'manifest.json'))
    guard(p)
    write(destination, state)


if __name__ == '__main__':
    {'prepare': prepare, 'preflight': preflight, 'run': run, 'audit': lambda: run(True)}[sys.argv[1]]()
