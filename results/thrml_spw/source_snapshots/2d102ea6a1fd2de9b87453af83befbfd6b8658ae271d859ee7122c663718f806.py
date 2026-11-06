"""Freeze the already specified heldout experiment after complete eligibility.

Reads metadata only. Never opens heldout tokens for model evaluation.
"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from research.thrml_spw.lfm3.holdout_batch_layout import batch_plan
from research.thrml_spw.lfm3.holdout_native_case_v2 import STAGES, validate_contract
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
        'adaptation_pilot/mean/module_aliases.json', 'holdout_native_statistics_v2_protocol.json',
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
    names = ['holdout_native_case_v2.py','verify_holdout_native_case_v2.py',
        'holdout_batch_layout.py','holdout_frontend_v2.py','holdout_treatment_selection_v2.py',
        'holdout_treatment_selection.py','count_budget_continuation.py','lossless_logits.py',
        'program_export.py','counted_programmed_worker.py','counted_bank.py']
    analysis = json.loads((root/'holdout_native_statistics_v2_protocol.json').read_text())
    names += list(analysis['sources_sha256'])
    sources = {name: sha(source_dir/name) for name in sorted(set(names))}
    for name,digest in analysis['sources_sha256'].items():assert sources[name]==digest
    for folder in ['attention_worker_cpu_preflight_v2','attention_worker_gpu_preflight_v2','attention_worker_h100_preflight_v2']:
        audit = json.loads((root/folder/'independent_audit.json').read_text())
        assert audit['status']=='all_four_attention_expanded_public_states_int64_counters_F64_readouts_and_receipts_verified'
        assert audit['results_sha256']==sha(root/folder/'results.json')
        if folder!='attention_worker_cpu_preflight_v2':
            assert audit['CPU_GPU_public_state_bit_differences']==audit['CPU_GPU_output_bitpattern_differences']==0
        inputs.update({folder+'/protocol.json',folder+'/results.json',folder+'/independent_audit.json'})
    geometry = Path('results/thrml_spw/broad_verification_20261105/model_geometry_v3/results.json')
    g = json.loads(geometry.read_text())
    assert g['status']=='two_actual_model_forwards_all_287_modules_and_rank4_norm_frontend_verified'
    for name,digest in g['sources_sha256'].items():assert sha(source_dir/name)==digest
    exposure = Path('results/thrml_spw/broad_verification_20261105/prior_holdout_exposure.json')
    assert sha(exposure)==analysis['exposure_ledger_sha256']
    auxiliary = [Path('research/thrml_spw') / n for n in
        ['native.py', 'controlled_positive_native.py', 'gpu_environment.py']]
    auxiliary += [Path('data/thrml_spw/lfm3_1p7b/checkpoint') / n for n in
        ['model.safetensors', 'config.json', 'modeling_lfm3.py', 'configuration_lfm3.py', 'lfm3_conv.py']]
    auxiliary += [geometry, exposure, Path('results/thrml_spw/broad_verification_20261105/statistics_v2_preflight/results.json')]
    previous = set()
    for n in ['counted_mean_validation_protocol.json', 'counted_mean_t4096_protocol.json', 'counted_mean_t8192_protocol.json']:
        previous.update(json.loads((root / n).read_text())['seeds'])
    seeds = list(range(202611150, 202611155))
    assert not previous.intersection(seeds)
    p = {
        'status': 'complete_execution_and_statistics_protocol_frozen_before_scoring',
        'recorded_utc': datetime.now(timezone.utc).isoformat(),
        'samples': selected['samples'], 'pbits_per_parameter': 1, 'arm': 'mean',
        'seeds': seeds, 'stages': STAGES, 'targets': 10240, 'articles': 40,
        'field_bits': 12, 'activation_bits': 12, 'batch_plan': batch_plan(manifest),
        'eligibility_policy_sha256': sha(policy_path),
        'statistics_protocol_sha256': sha(root / 'holdout_native_statistics_v2_protocol.json'),
        'independent_case_verifier_sha256': sources['verify_holdout_native_case_v2.py'],
        'sources_sha256': sources,
        'inputs_sha256': {n: sha(root / n) for n in sorted(inputs)},
        'auxiliary_inputs_sha256': {str(path): sha(path) for path in auxiliary},
        'execution': {'gpu': 'H100!', 'max_parallel_cases': 4, 'sampling_retries': 0,
            'case_timeout_seconds': 86400, 'first_seed_audited_before_remaining_four': True},
        'controls': 'All heldout BF16 and same-fit BF16 controls computed on H100 in every case; exact deterministic output checks against first heldout H100 seed. No local/cloud seed pooling.',
        'scope': 'Repaired-rank, versioned complete evaluation after a preserved failed run. Full40andpreviouslyunexposed38articles are separately evaluated. Retain every result and failed case. No threshold tuning, retry, numerical BF16 equivalence, autoregressive or physical performance claim.'}
    for seed in seeds:
        validate_contract(p, selected, p['batch_plan'], seed)
    destination = root / 'holdout_native_v2_protocol.json'
    with destination.open('x') as f:
        json.dump(p, f, indent=2)
        f.write('\n')
    files = {str(root / name): {'sha256': digest, 'bytes': (root / name).stat().st_size}
        for name, digest in p['inputs_sha256'].items()}
    for path in [geometry, exposure, Path('results/thrml_spw/broad_verification_20261105/statistics_v2_preflight/results.json')]:
        files[str(path)]={'sha256':sha(path),'bytes':path.stat().st_size}
    files[str(destination)] = {'sha256': sha(destination), 'bytes': destination.stat().st_size}
    out = Path('results/thrml_spw/broad_verification_20261105/holdout_assets_v2.json')
    with out.open('x') as f:
        json.dump({'status': 'frozen_heldout_inputs_after_complete_eligibility', 'files': files,
            'execution_protocol_sha256': sha(destination)}, f, indent=2)
        f.write('\n')
    print(json.dumps({'status': 'holdout_execution_frozen', 'protocol_sha256': sha(destination),
        'samples': selected['samples'], 'seeds': seeds}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
