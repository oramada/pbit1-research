"""Test new audit arithmetic on accepted validation evidence, never heldout text."""
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from .verify_holdout_native_case import resource_audit, independent_losses


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(16*1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def main():
    torch.set_num_threads(2)
    root = Path('results/thrml_spw/lfm3_1p7b')
    old = root/'counted_mean_t8192/mean/202610600'
    out = root/'holdout_case_auditor_preflight'
    out.mkdir(exist_ok=False)
    audit = json.loads((old/'independent_audit.json').read_text())
    result = json.loads((old/'results.json').read_text())
    assert audit['status'] == 'all_direct_program_coverage_losses_and_resource_receipts_verified'
    assert audit['results_sha256'] == sha(old/'results.json')
    ids = np.load(root/'adaptation_data/tokens.npz')['validation_tokens']
    assert ids.shape == (32, 65)
    ids_by_batch = [ids[i:i+8] for i in range(0, 32, 8)]
    shapes = {v['name']: v['shape'] for v in json.loads((root/'meta_inventory.json').read_text())['parameters']}
    aliases = json.loads((root/'adaptation_pilot/mean/module_aliases.json').read_text())
    programs = {v['name']: v['program_sha256'] for v in json.loads((root/'adaptation_pilot/mean/program_package.json').read_text())['programs']}
    counts = resource_audit(result['operator_calls'], ids_by_batch, shapes, aliases, programs, 8192)
    for key in ['retained_scheduled_evaluations', 'warmup_scheduled_evaluations', 'sampled_coordinates_across_uses',
                'max_temporary_weight_count_bytes', 'max_simultaneous_sampler_state_slots']:
        assert counts[key] == audit[key]
    rejected = []
    for name in ['missing_call', 'duplicate_module', 'wrong_order', 'wrong_program', 'retained', 'warmup',
                 'counter_bytes', 'reference_weight', 'activation_precision', 'embedding_rows']:
        calls = copy.deepcopy(result['operator_calls'])
        if name == 'missing_call': calls.pop()
        elif name == 'duplicate_module': calls[1] = copy.deepcopy(calls[0])
        elif name == 'wrong_order': calls[0], calls[1] = calls[1], calls[0]
        elif name == 'wrong_program': calls[0]['program_sha256'] = 'forged'
        elif name == 'retained': next(v for v in calls if v['stage']=='native')['retained_evaluations'] -= 1
        elif name == 'warmup': next(v for v in calls if v['stage']=='native')['warmup_evaluations'] = 0
        elif name == 'counter_bytes': next(v for v in calls if v['stage']=='native')['simulation_count_workspace_bytes'] = 0
        elif name == 'reference_weight': calls[0]['reference_weights_supplied'] = True
        elif name == 'activation_precision': calls[0]['activation_bits'] = 8
        else: calls[0]['embedding_rows'] = []
        try:
            resource_audit(calls, ids_by_batch, shapes, aliases, programs, 8192)
        except AssertionError:
            rejected.append(name)
        else:
            raise AssertionError('Corrupted receipt accepted: '+name)
    saved = old/'batch0.npz'
    assert sha(saved) == result['outputs'][0]['sha256']
    with np.load(saved) as a:
        excerpt = a['native'][0:1, :18].copy()
    targets = ids[0:1, :18].copy()
    actual = independent_losses(excerpt, targets)
    metric = next(v for v in result['metrics'] if v['batch']==0 and v['stage']=='native')
    expected = np.asarray(metric['token_NLL'], dtype=np.float64)[0:1, :17]
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    raw_path = out/'accepted_validation_excerpt.npz'
    with raw_path.open('xb') as f:
        np.savez_compressed(f, logits=excerpt, input_ids=targets, independent_losses=actual, producer_losses=expected)
    names = ['verify_holdout_native_case.py', 'preflight_verify_holdout_native_case.py',
        'holdout_batch_layout.py', 'holdout_treatment_selection_v2.py',
        'holdout_treatment_selection.py', 'count_budget_continuation.py', 'lossless_logits.py']
    report = {'status': 'accepted_validation_receipts_raw_loss_excerpt_and_ten_corruptions_verified',
        'sources_sha256': {n: sha(Path(__file__).with_name(n)) for n in names},
        'reference_audit_sha256': sha(old/'independent_audit.json'),
        'reference_results_sha256': sha(old/'results.json'), 'reference_raw_batch_sha256': sha(saved),
        'resources': counts, 'weighted_modules_per_stage_batch': 287, 'unique_parameter_coordinates': 1728801536,
        'invalid_receipts_rejected': rejected, 'excerpt_sha256': sha(raw_path),
        'excerpt_targets': 17, 'excerpt_vocabulary': 128000,
        'max_absolute_loss_difference': float(np.abs(actual-expected).max()),
        'scope': 'Core independent auditor logic replayed on already accepted/consumed validation evidence. Not a new random seed or model evaluation. Main reserved-case audit I/O, full scoring protocol, analysis and actual reserved inference remain unexecuted. No reserved token arrays or predictions opened.'}
    with (out/'results.json').open('x') as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report))


if __name__ == '__main__':
    main()
