"""Early full-domain learned-mean control check, without labels or task metrics."""
import datetime
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

from results.thrml_spw.fashion_cnn_native.probability_cohort.common import ROOT, OUT as PARENT, BASE, checkpoint
from results.thrml_spw.fashion_cnn_native.probability_cohort.verify import reference

OUT = ROOT/'probability_control_review_evidence'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, obj):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2)+'\n')
    tmp.replace(path)


def guard(p):
    for mapping in [p['source_sha256'], p['input_sha256']]:
        for file, expected in mapping.items():
            assert sha(file) == expected, file


def prepare():
    OUT.mkdir(exist_ok=True)
    assert not (OUT/'protocol.json').exists()
    parent = json.loads((PARENT/'protocol.json').read_text())
    controls = json.loads((PARENT/'controls.json').read_text())
    assert controls['complete'] and controls['mean_logits_sha256'] == sha(PARENT/'mean_logits.npz')
    p = {'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
         'source_sha256': {**parent['source_sha256'], __file__: sha(__file__)},
         'input_sha256': {**parent['input_sha256'], **{str(path): sha(path) for path in [PARENT/'protocol.json', PARENT/'controls.json', PARENT/'mean_logits.npz']}},
         'models': 5, 'methods': 2, 'images': 10000, 'batch': 128,
         'rtol': 2e-12, 'atol': 2e-11,
         'scope': 'Read-only early check of all10nominal learned-mean controls on fixed previously consumed inputs. Independently reconstructed fields/ranges and TorchF64graph versus producerNumPyF64graph. No labels/loss/accuracy or sampler calls. Frozen final full-cohort verifier still runs unchanged after all native/public cases.'}
    guard(p)
    write(OUT/'protocol.json', p)


def run():
    p = json.loads((OUT/'protocol.json').read_text())
    guard(p)
    assert not (OUT/'completion.json').exists()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    x = np.load(BASE/'test_data.npz')['inputs']
    originals = json.loads((ROOT/'training/manifest.json').read_text())['rows']
    saved = np.load(PARENT/'mean_logits.npz')
    rows = []
    for method in range(2):
        for mi in range(5):
            model = checkpoint(mi, method)
            original = np.load(originals[mi]['path'])
            actual = saved[f'method{method}_model{mi}']
            independent = np.concatenate([reference(x[first:first+p['batch']], model, original) for first in range(0, 10000, p['batch'])])
            np.testing.assert_allclose(actual, independent, rtol=p['rtol'], atol=p['atol'])
            np.testing.assert_array_equal(actual.argmax(1), independent.argmax(1))
            rows.append({'model': mi, 'method': method, 'logit_values_checked': actual.size,
                         'predictions_checked': len(actual), 'max_logit_error': float(np.max(abs(actual-independent))),
                         'all_predictions_identical': True})
            print(json.dumps(rows[-1]), flush=True)
    assert len(rows) == 10 and sum(r['logit_values_checked'] for r in rows) == 1000000
    guard(p)
    write(OUT/'evidence_audit.json', {'passed': True, 'rows': rows, 'source_sha256': sha(__file__),
                                   'controls_checked': 10, 'full_logit_values_checked': 1000000,
                                   'predictions_checked': 100000, 'new_native_updates': 0,
                                   'labels_read': False, 'task_metrics_computed': False,
                                   'scope': p['scope']})
    write(OUT/'completion.json', {'status': 'all10full_domain_learned_mean_controls_checked_goal_active',
                                 'artifact_sha256': {str(path): sha(path) for path in [OUT/'protocol.json', OUT/'evidence_audit.json']},
                                 'new_native_updates': 0, 'task_metrics_computed': False})


if __name__ == '__main__':
    {'prepare': prepare, 'run': run}[sys.argv[1]]()
