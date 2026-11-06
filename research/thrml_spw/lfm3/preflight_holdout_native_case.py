"""Prospective runner contract and loss arithmetic; no model/heldout scoring."""
import copy
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from .holdout_batch_layout import batch_plan
from .holdout_native_case import STAGES, validate_contract, token_nll


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path('results/thrml_spw/lfm3_1p7b')
    out = root / 'holdout_native_case_preflight'
    out.mkdir(exist_ok=False)
    manifest_path = root / 'holdout_data/manifest.json'
    plan = batch_plan(json.loads(manifest_path.read_text()))
    protocol = {'status': 'complete_execution_and_statistics_protocol_frozen_before_scoring',
        'samples': 8192, 'seeds': [202610800 + i for i in range(5)], 'stages': STAGES,
        'targets': 10240, 'articles': 40, 'field_bits': 12, 'activation_bits': 12,
        'batch_plan': plan, 'statistics_protocol_sha256': 'synthetic-only',
        'independent_case_verifier_sha256': 'synthetic-only'}
    selection = {'status': 'eligible_after_full_execution_protocol_freeze',
        'pbits_per_parameter': 1, 'arm': 'mean', 'samples': 8192, 'score_holdout': False}
    assert validate_contract(protocol, selection, plan, 202610800)
    rejected = []
    for name in ['partial_validation', 'unfrozen_protocol', 'four_seeds', 'duplicate_seed', 'invalid_seed',
                 'wrong_budget', 'short_only', 'missing_controls', 'changed_plan', 'no_analysis', 'no_verifier', 'multibit']:
        p, s = copy.deepcopy(protocol), copy.deepcopy(selection)
        if name == 'partial_validation': s['status'] = 'pending_complete_T8192_cohort'
        elif name == 'unfrozen_protocol': p['status'] = 'draft'
        elif name == 'four_seeds': p['seeds'].pop()
        elif name == 'duplicate_seed': p['seeds'][1] = p['seeds'][0]
        elif name == 'invalid_seed': p['seeds'][1] = -1
        elif name == 'wrong_budget': p['samples'] = 4096
        elif name == 'short_only': p['targets'] = 6144
        elif name == 'missing_controls': p['stages'] = ['native']
        elif name == 'changed_plan': p['batch_plan'].pop()
        elif name == 'no_analysis': p['statistics_protocol_sha256'] = ''
        elif name == 'no_verifier': p['independent_case_verifier_sha256'] = ''
        else: s['pbits_per_parameter'] = 4
        try:
            validate_contract(p, s, plan, 202610800)
        except ValueError:
            rejected.append(name)
        else:
            raise AssertionError('Invalid execution contract accepted: ' + name)
    rng = np.random.default_rng(202610702)
    loss_records = []
    for i, shape in enumerate([(8, 65, 43), (4, 257, 43)]):
        logits = (rng.standard_normal(shape) * 10).astype(np.float32)
        logits[0, 0, 0], logits[0, 0, 1] = 1200, -1200
        ids = rng.integers(0, shape[2], shape[:2], dtype=np.int64)
        actual = token_nll(logits, ids)
        independent = np.empty(actual.shape)
        for row in range(shape[0]):
            for pos in range(shape[1]-1):
                values = list(map(float, logits[row, pos]))
                maximum = max(values)
                independent[row, pos] = maximum + math.log(math.fsum(math.exp(v-maximum) for v in values)) - values[int(ids[row, pos+1])]
        np.testing.assert_allclose(actual, independent, rtol=0, atol=2e-12)
        path = out / f'synthetic_losses{i}.npz'
        with path.open('xb') as f:
            np.savez_compressed(f, logits=logits, input_ids=ids, chunked_losses=actual, scalar_oracle_losses=independent)
        loss_records.append({'path': str(path), 'sha256': sha(path), 'shape': list(shape),
            'targets': actual.size, 'max_absolute_loss_difference': float(np.abs(actual-independent).max())})
    ids = np.zeros((1, 17), dtype=np.int64)
    logits = np.zeros((1, 17, 43), dtype=np.float32)
    np.testing.assert_allclose(token_nll(logits, ids), math.log(43), rtol=0, atol=1e-15)
    for name, ll, tt in [('negative_id', logits, ids-1), ('large_id', logits, ids+43),
            ('floating_id', logits, ids.astype(float)), ('wrong_dtype', logits.astype(float), ids),
            ('wrong_shape', logits[:, :-1], ids), ('empty_sequence', logits[:, :0], ids[:, :0])]:
        try:
            token_nll(ll, tt)
        except ValueError:
            rejected.append(name)
        else:
            raise AssertionError('Invalid loss inputs accepted')
    source = Path(__file__)
    names = ['preflight_holdout_native_case.py', 'holdout_native_case.py', 'holdout_batch_layout.py',
        'holdout_frontend.py', 'holdout_treatment_selection_v2.py', 'holdout_treatment_selection.py',
        'count_budget_continuation.py', 'lossless_logits.py', 'program_export.py']
    result = {'status': 'prospective_runner_contract_rejections_and_chunked_losses_verified_without_scoring',
        'sources_sha256': {n: sha(source.with_name(n)) for n in names},
        'manifest_sha256': sha(manifest_path), 'invalid_inputs_rejected': rejected,
        'synthetic_loss_cases': loss_records,
        'scope': 'Synthetic pure contract checks and stable chunked loss arithmetic only. Main runner, THRML worker, full model, eligibility file I/O, source/input freeze coverage and future independent scoring verifier are not integration-tested here. No reserved token arrays or predictions opened. Draft runner is not a frozen execution protocol or an accepted inference implementation.'}
    with (out / 'results.json').open('x') as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
