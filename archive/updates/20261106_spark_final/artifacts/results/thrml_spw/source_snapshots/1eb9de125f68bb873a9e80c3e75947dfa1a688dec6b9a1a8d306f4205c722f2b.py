"""Cold-process same-code reproduction of all probability fine-tuning fits."""
import datetime
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path('results/thrml_spw/cifar_cnn_native')
PARENT = ROOT/'probability_finetune_evidence'
OUT = ROOT/'probability_training_reproduction_evidence'


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


def prepare():
    OUT.mkdir(exist_ok=True)
    assert not (OUT/'protocol.json').exists()
    fine_protocol = json.loads((PARENT/'protocol.json').read_text())
    closed = json.loads((PARENT/'completion.json').read_text())
    for path, expected in closed['artifact_sha256'].items():
        assert sha(path) == expected, path
    assert json.loads((PARENT/'evidence_audit.json').read_text())['passed']
    p = {'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
         'source_sha256': {**fine_protocol['source_sha256'], __file__: sha(__file__)},
         'input_sha256': {**fine_protocol['input_sha256'], **closed['artifact_sha256'],
                          str(PARENT/'completion.json'): sha(PARENT/'completion.json')},
         'fits': 10, 'epochs': 80, 'optimizer_steps': 14080,
         'validation_logits': 2000000, 'minimum_free_bytes': 50000000,
         'extra_payload_bound_bytes': 20000000,
         'comparison': 'Require exact epoch-history bytes, selected epochs/CE and every exported array dtype/shape/raw byte. No tolerance or outcome selection. ZIP container bytes need not match.',
         'scope': 'Fresh process repeats original frozen training implementation/data/splits/seeds/environment in isolated output directory. Only module output-path global redirected; source files and original results unchanged. Same-code reproduction, not independently implemented optimizer/gradient or new MC/generalization replicas. Training forwards remain Torch surrogates, not THRML calls.'}
    guard(p)
    write(OUT/'protocol.json', p)


def run():
    from results.thrml_spw.cifar_cnn_native.probability_finetune import train
    p = json.loads((OUT/'protocol.json').read_text())
    guard(p)
    assert not (OUT/'pipeline.json').exists()
    assert shutil.disk_usage('.').free >= p['minimum_free_bytes']
    target = OUT/'training'
    target.mkdir()
    for name in ['protocol.json', 'selfcheck.json']:
        (target/name).write_bytes((PARENT/name).read_bytes())
    write(OUT/'pipeline.json', {'status': 'reproducing_original_training_implementation', 'new_native_updates': 0})
    # Runtime path redirection is local to this fresh process; no source edits,
    # new seeds, parameter changes or original-output mutation are performed.
    train.OUT = target
    train.run()
    guard(p)
    write(OUT/'pipeline.json', {'status': 'training_reproduced_comparison_pending', 'new_native_updates': 0})
    original = json.loads((PARENT/'manifest.json').read_text())
    repeated = json.loads((target/'manifest.json').read_text())
    assert original['complete'] and repeated['complete'] and len(repeated['rows']) == 10
    old = {(r['model'], r['method']): r for r in original['rows']}
    assert set(old) == {(r['model'], r['method']) for r in repeated['rows']}
    rows = []
    arrays_checked = 0
    values_checked = 0
    for row in repeated['rows']:
        prior = old[row['model'], row['method']]
        assert row['selected_epoch'] == prior['selected_epoch']
        assert row['validation_CE'] == prior['validation_CE']
        history = f"model{row['model']}_{row['method']}_history.json"
        assert (target/history).read_bytes() == (PARENT/history).read_bytes()
        assert sha(row['path']) == row['sha256'] and sha(prior['path']) == prior['sha256']
        with np.load(prior['path']) as a, np.load(row['path']) as b:
            assert set(a.files) == set(b.files)
            for name in a.files:
                x, y = a[name], b[name]
                assert x.dtype == y.dtype and x.shape == y.shape
                assert x.tobytes() == y.tobytes(), (row['model'], row['method'], name)
                arrays_checked += 1
                values_checked += x.size
        rows.append({'model': row['model'], 'method': row['method'], 'selected_epoch': row['selected_epoch'],
                     'all_export_arrays_byte_exact': True, 'all_epoch_history_bytes_exact': True,
                     'original_checkpoint_sha256': prior['sha256'], 'reproduced_checkpoint_sha256': row['sha256']})
    audit = {'passed': True, 'recorded_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
             'rows': rows, 'array_count_checked': arrays_checked, 'array_values_checked': values_checked,
             'original_training_optimizer_steps': 14080, 'additional_reproduction_optimizer_steps': 14080,
             'original_plus_reproduction_optimizer_steps': 28160,
             'all_selected_validation_logit_values_byte_exact': 2000000,
             'original_manifest_sha256': sha(PARENT/'manifest.json'), 'reproduced_manifest_sha256': sha(target/'manifest.json'),
             'source_sha256': sha(__file__), 'new_native_updates': 0, 'scope': p['scope']}
    write(OUT/'evidence_audit.json', audit)
    paths = [OUT/'protocol.json', OUT/'evidence_audit.json', target/'protocol.json', target/'manifest.json']
    paths += [Path(row['path']) for row in repeated['rows']]
    paths += [target/f"model{row['model']}_{row['method']}_history.json" for row in repeated['rows']]
    guard(p)
    write(OUT/'completion.json', {'status': 'all10RGBprobability_training_fits_byte_reproduced_goal_active',
                                 'artifact_sha256': {str(path): sha(path) for path in paths},
                                 'additional_optimizer_steps': 14080, 'new_native_updates': 0})
    write(OUT/'pipeline.json', {'status': 'all10RGBprobability_training_fits_byte_reproduced_goal_active', 'new_native_updates': 0})
    print('All10training fits, full epoch histories and all selected exports byte-reproduced', flush=True)


if __name__ == '__main__':
    {'prepare': prepare, 'run': run}[sys.argv[1]]()
