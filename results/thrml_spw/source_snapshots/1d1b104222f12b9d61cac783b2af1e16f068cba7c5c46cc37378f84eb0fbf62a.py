"""Freeze the already specified heldout experiment after complete eligibility.

Reads metadata only. Never opens heldout tokens for model evaluation.
"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from research.thrml_spw.lfm3.holdout_batch_layout import batch_plan
from research.thrml_spw.lfm3.holdout_native_case import STAGES, validate_contract
from research.thrml_spw.lfm3.holdout_treatment_selection_v2 import select_treatment


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(16 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    root = Path('results/thrml_spw/lfm3_1p7b')
    policy_path = root / 'holdout_treatment_selection_v2_policy.json'
    policy = json.loads(policy_path.read_text())
    bundles = []
    inputs = {
        'holdout_data/manifest.json', 'holdout_data/tokens.npz',
        'holdout_data/independent_audit.json', 'holdout_treatment_selection_v2_policy.json',
        'adaptation_pilot/mean/program_package.json', 'adaptation_pilot/mean/independent_audit.json',
        'adaptation_pilot/mean/module_aliases.json', 'holdout_native_statistics_protocol.json',
        'counted_mean_validation_protocol.json', 'counted_mean_t4096_protocol.json',
        'counted_mean_t8192_protocol.json', 'meta_inventory.json',
        'counted_mean_validation/mean/202610350/results.json',
        'counted_mean_validation/mean/202610350/independent_audit.json',
    }
    for name in ['counted_mean_validation', 'counted_mean_t4096', 'counted_mean_t8192']:
        sp, ap = root / name / 'statistics.json', root / name / 'statistics_audit.json'
        if not sp.exists() or not ap.exists():
            print(json.dumps({'status': 'waiting_complete_validation', 'condition': name}))
            return 2
        bundles.append({'statistics': json.loads(sp.read_text()),
            'audit': json.loads(ap.read_text()), 'statistics_sha256': sha(sp)})
        inputs.update({name + '/statistics.json', name + '/statistics_audit.json'})
    selected = select_treatment(*bundles, policy)
    if selected['status'] != 'eligible_after_full_execution_protocol_freeze':
        print(json.dumps(selected))
        return 3
    manifest = json.loads((root / 'holdout_data/manifest.json').read_text())
    assert sha(root / 'holdout_data/manifest.json') == policy['holdout_manifest_sha256']
    assert sha(root / 'holdout_data/tokens.npz') == manifest['tokens_sha256']
    source_dir = Path('research/thrml_spw/lfm3')
    sources = {p.name: sha(p) for p in sorted(source_dir.glob('*.py'))}
    auxiliary = [Path('research/thrml_spw') / n for n in
        ['native.py', 'controlled_positive_native.py', 'gpu_environment.py']]
    auxiliary += [Path('data/thrml_spw/lfm3_1p7b/checkpoint') / n for n in
        ['model.safetensors', 'config.json', 'modeling_lfm3.py', 'configuration_lfm3.py', 'lfm3_conv.py']]
    previous = set()
    for n in ['counted_mean_validation_protocol.json', 'counted_mean_t4096_protocol.json', 'counted_mean_t8192_protocol.json']:
        previous.update(json.loads((root / n).read_text())['seeds'])
    seeds = list(range(202610800, 202610805))
    assert not previous.intersection(seeds)
    p = {
        'status': 'complete_execution_and_statistics_protocol_frozen_before_scoring',
        'recorded_utc': datetime.now(timezone.utc).isoformat(),
        'samples': selected['samples'], 'pbits_per_parameter': 1, 'arm': 'mean',
        'seeds': seeds, 'stages': STAGES, 'targets': 10240, 'articles': 40,
        'field_bits': 12, 'activation_bits': 12, 'batch_plan': batch_plan(manifest),
        'eligibility_policy_sha256': sha(policy_path),
        'statistics_protocol_sha256': sha(root / 'holdout_native_statistics_protocol.json'),
        'independent_case_verifier_sha256': sources['verify_holdout_native_case.py'],
        'sources_sha256': sources,
        'inputs_sha256': {n: sha(root / n) for n in sorted(inputs)},
        'auxiliary_inputs_sha256': {str(path): sha(path) for path in auxiliary},
        'execution': {'gpu': 'H100!', 'max_parallel_cases': 4, 'sampling_retries': 0,
            'case_timeout_seconds': 86400, 'first_seed_audited_before_remaining_four': True},
        'controls': 'All heldout BF16 and same-fit BF16 controls computed on H100 in every case; exact deterministic output checks against first heldout H100 seed. No local/cloud seed pooling.',
        'scope': 'Complete reserved teacher-forced evaluation under the previously frozen analysis. Retain every result and failed case. No threshold tuning, retry, numerical BF16 equivalence, autoregressive or physical performance claim.'}
    for seed in seeds:
        validate_contract(p, selected, p['batch_plan'], seed)
    destination = root / 'holdout_native_protocol.json'
    with destination.open('x') as f:
        json.dump(p, f, indent=2)
        f.write('\n')
    files = {str(root / name): {'sha256': digest, 'bytes': (root / name).stat().st_size}
        for name, digest in p['inputs_sha256'].items()}
    files[str(destination)] = {'sha256': sha(destination), 'bytes': destination.stat().st_size}
    out = Path('results/thrml_spw/modal_h100_v1/holdout_assets_v1.json')
    with out.open('x') as f:
        json.dump({'status': 'frozen_heldout_inputs_after_complete_eligibility', 'files': files,
            'execution_protocol_sha256': sha(destination)}, f, indent=2)
        f.write('\n')
    print(json.dumps({'status': 'holdout_execution_frozen', 'protocol_sha256': sha(destination),
        'samples': selected['samples'], 'seeds': seeds}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
