"""Prospective independent raw-loss/coverage/resource auditor for reserved cases.

Main entry point requires a frozen complete protocol and accepted eligibility.
Core arithmetic can be checked on already consumed validation artifacts.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from .holdout_batch_layout import batch_plan, assemble_losses
from .holdout_treatment_selection_v2 import select_treatment
from .lossless_logits import read_logits


STAGES = ['BF16', 'same_fit_BF16', 'nominal_controls', 'native']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(16*1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def independent_losses(logits, ids):
    assert logits.dtype == np.float32 and logits.ndim == 3 and logits.shape[:2] == ids.shape
    assert ids.dtype == np.int64 and ids.shape[1] >= 2
    assert ids.min() >= 0 and ids.max() < logits.shape[-1] and np.isfinite(logits).all()
    result = np.empty((ids.shape[0], ids.shape[1]-1), dtype=np.float64)
    for row in range(ids.shape[0]):
        for start in range(0, ids.shape[1]-1, 8):
            stop = min(start+8, ids.shape[1]-1)
            values = torch.from_numpy(logits[row, start:stop].copy()).double()
            targets = torch.from_numpy(ids[row, start+1:stop+1].copy())
            result[row, start:stop] = F.cross_entropy(values, targets, reduction='none').numpy()
    assert np.isfinite(result).all()
    return result


def resource_audit(calls, ids_by_batch, shapes, aliases, program_hashes, samples):
    """Independently derive every receipt from canonical shapes and input IDs."""
    assert type(samples) is int and samples > 0
    assert len(calls) == 2*len(ids_by_batch)*len(aliases)
    assert set(shapes) == set(aliases.values()) == set(program_hashes)
    assert set(x['stage'] for x in calls) == {'nominal_controls', 'native'}
    assert set(x['batch'] for x in calls) == set(range(len(ids_by_batch)))
    expected_order = None
    retained = warmup = coordinates = maximum_counts = maximum_states = 0
    batches = []
    for batch, ids in enumerate(ids_by_batch):
        br = bw = 0
        for stage in ['nominal_controls', 'native']:
            group = [x for x in calls if x['batch'] == batch and x['stage'] == stage]
            assert len(group) == len(aliases) and {x['module'] for x in group} == set(aliases)
            order = [x['module'] for x in group]
            if expected_order is None:
                expected_order = order
            assert order == expected_order
            for index, x in enumerate(group):
                name = aliases[x['module']]
                shape = shapes[name]
                kind = ('embedding' if x['module'] == 'model.embed_tokens' else
                        'linear' if len(shape) == 2 else 'conv' if len(shape) == 3 else 'elementwise')
                assert x['canonical_program'] == name and x['program_sha256'] == program_hashes[name]
                assert x['kind'] == kind and x['call'] == 287+batch*287+index
                if kind == 'embedding':
                    rows = np.unique(ids).tolist()
                    assert x['embedding_rows'] == rows
                    fanin, channels = 1, len(rows)*shape[1]
                elif kind == 'linear':
                    fanin, channels = shape[1], shape[0]
                    assert x['embedding_rows'] is None
                elif kind == 'conv':
                    assert shape[1:] == [1, 7]
                    fanin, channels = 7, shape[0]
                    assert x['embedding_rows'] is None
                else:
                    fanin, channels = 1, shape[0]
                    assert x['embedding_rows'] is None
                size = min(128, fanin)
                assert fanin % size == 0 and 2047*size*samples < 2**31
                tiles = (channels+63)//64
                padded = fanin*tiles*64
                assert x['weight_coordinates'] == fanin*channels and x['padding_nodes'] == padded-fanin*channels
                assert x['ok'] and x['finite'] and x['program_codes_used_directly'] and not x['reference_weights_supplied']
                assert x['field_bits'] == x['activation_bits'] == 12 and x['range_metadata_dtype'] == 'float32'
                assert x['simulation_observer'] == 'per_weight_counts_then_exact_integer_contraction'
                assert x['tile_lanes_limit'] == 4 and x['max_simultaneous_tile_lanes'] == min(4, tiles)
                assert x['tile_batches'] == (tiles+3)//4
                native = stage == 'native'
                assert x['retained_evaluations'] == (padded*samples if native else 0)
                assert x['warmup_evaluations'] == (padded if native else 0)
                assert x['simulation_count_workspace_bytes'] == (4*min(4,tiles)*fanin*64 if native else 0)
                assert x['max_simultaneous_sampler_state_slots'] == min(4,tiles)*fanin*64
                if native:
                    br += padded*samples; bw += padded; coordinates += fanin*channels
                maximum_counts = max(maximum_counts, x['simulation_count_workspace_bytes'])
                maximum_states = max(maximum_states, x['max_simultaneous_sampler_state_slots'])
        retained += br; warmup += bw
        batches.append({'batch': batch, 'retained_scheduled_evaluations': br, 'warmup_scheduled_evaluations': bw})
    return {'retained_scheduled_evaluations': retained, 'warmup_scheduled_evaluations': warmup,
        'sampled_coordinates_across_uses': coordinates, 'max_temporary_weight_count_bytes': maximum_counts,
        'max_simultaneous_sampler_state_slots': maximum_states, 'per_batch': batches}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--protocol', type=Path, default=Path('results/thrml_spw/lfm3_1p7b/holdout_native_v2_protocol.json'))
    args = parser.parse_args()
    torch.set_num_threads(2)
    root = Path('results/thrml_spw/lfm3_1p7b')
    p = json.loads(args.protocol.read_text())
    assert p['status'] == 'complete_execution_and_statistics_protocol_frozen_before_scoring'
    assert sha(Path(__file__)) == p['independent_case_verifier_sha256'] == p['sources_sha256'][Path(__file__).name]
    assert {'meta_inventory.json', 'adaptation_pilot/mean/module_aliases.json',
        'counted_mean_validation/mean/202610350/results.json',
        'counted_mean_validation/mean/202610350/independent_audit.json'} <= set(p['inputs_sha256'])
    for name, digest in p['sources_sha256'].items():
        assert sha(Path(__file__).with_name(name)) == digest
    for name, digest in p['inputs_sha256'].items():
        assert sha(root/name) == digest
    for name, digest in p['auxiliary_inputs_sha256'].items():
        assert sha(name) == digest
    assert p['statistics_protocol_sha256'] == sha(root/'holdout_native_statistics_v2_protocol.json')
    policy_path = root/'holdout_treatment_selection_v2_policy.json'
    assert sha(policy_path) == p['eligibility_policy_sha256']
    policy = json.loads(policy_path.read_text())
    bundles = []
    for name in ['counted_mean_validation', 'counted_mean_t4096', 'counted_mean_t8192']:
        sp, ap = root/name/'statistics.json', root/name/'statistics_audit.json'
        bundles.append({'statistics': json.loads(sp.read_text()), 'audit': json.loads(ap.read_text()), 'statistics_sha256': sha(sp)})
    chosen = select_treatment(*bundles, policy)
    assert chosen['status'] == 'eligible_after_full_execution_protocol_freeze' and chosen['samples'] == p['samples']
    assert chosen['arm'] == 'mean' and chosen['pbits_per_parameter'] == 1
    assert len(p['seeds']) == len(set(p['seeds'])) == 5 and args.seed in p['seeds']
    assert p['stages'] == STAGES and p['targets'] == 10240 and p['articles'] == 40
    manifest_path = root/'holdout_data/manifest.json'
    assert sha(manifest_path) == policy['holdout_manifest_sha256']
    manifest = json.loads(manifest_path.read_text()); plan = batch_plan(manifest)
    assert p['batch_plan'] == plan
    out = root/'holdout_native_v2'/str(args.seed)
    r = json.loads((out/'results.json').read_text()); rp = json.loads((out/'protocol.json').read_text())
    assert r['status'] == 'full_reserved_native_case_produced_pending_independent_audit'
    assert r['seed'] == rp['seed'] == args.seed and r['samples'] == rp['samples'] == chosen['samples']
    assert r['shared_protocol_sha256'] == rp['shared_protocol_sha256'] == sha(args.protocol)
    assert rp['selection'] == chosen and r['pbits_per_parameter'] == 1
    assert r['unique_parameter_coordinates'] == 1728801536 and r['unique_programs'] == 286 and r['weighted_modules'] == 287
    assert r['targets'] == 10240 and r['articles'] == 40
    assert r['worker_ready']['ready'] and any('cuda' in x.lower() or 'gpu' in x.lower() for x in r['worker_ready']['devices'])
    assert p['field_bits'] == p['activation_bits'] == 12
    package_dir = root/'adaptation_pilot/mean'
    assert sha(package_dir/'program_package.json') == rp['package_sha256'] == policy['adapted_mean_package_sha256']
    package = json.loads((package_dir/'program_package.json').read_text())
    records = {x['name']: x for x in package['programs']}
    aliases = json.loads((package_dir/'module_aliases.json').read_text())
    shapes = {x['name']: x['shape'] for x in json.loads((root/'meta_inventory.json').read_text())['parameters']}
    assert len(shapes) == len(records) == 286 and len(aliases) == 287
    assert sum(math.prod(shape) for shape in shapes.values()) == 1728801536
    for module, name in aliases.items():
        assert name == ('model.embed_tokens.weight' if module == 'lm_head' else module+'.weight')
    for name, rec in records.items():
        assert rec['shape'] == shapes[name] and sha(rec['program']) == rec['program_sha256']
    master = root/'counted_mean_validation/mean/202610350'
    ma = json.loads((master/'independent_audit.json').read_text())
    assert sha(master/'independent_audit.json') == policy['master_control_audit_sha256']
    assert ma['status'] == 'all_direct_program_coverage_losses_and_resource_receipts_verified'
    assert sha(master/'results.json') == ma['results_sha256']
    assert r['same_fit_BF16_weight_hashes'] == json.loads((master/'results.json').read_text())['same_fit_BF16_weight_hashes']
    assert sha(root/'holdout_data/tokens.npz') == manifest['tokens_sha256']
    with np.load(root/'holdout_data/tokens.npz', allow_pickle=False) as arrays:
        inputs = {name: arrays[name+'_tokens'].copy() for name in ['short', 'long']}
    ids_by_batch = [inputs[s['cohort']][s['window_start']:s['window_stop']] for s in plan]
    resources = resource_audit(r['operator_calls'], ids_by_batch, shapes, aliases,
        {name: rec['program_sha256'] for name, rec in records.items()}, p['samples'])
    assert len(r['outputs']) == len(r['metrics']) == 4*len(plan) == 64
    losses = {stage: [] for stage in STAGES}
    for batch, spec in enumerate(plan):
        ids = ids_by_batch[batch]
        assert list(ids.shape) == spec['input_shape'] and ids.dtype == np.int64
        for stage in STAGES:
            outputs = [x for x in r['outputs'] if x['batch'] == batch and x['stage'] == stage]
            metrics = [x for x in r['metrics'] if x['batch_index'] == batch and x['stage'] == stage]
            assert len(outputs) == len(metrics) == 1
            item, metric = outputs[0], metrics[0]
            path = out/f'batch{batch}_{stage}.npz'
            assert Path(item['path']) == path and sha(path) == item['sha256']
            logits = read_logits(path)
            assert logits.shape == (*ids.shape, 128000) and logits.dtype == np.float32 and np.isfinite(logits).all()
            assert item['shape'] == list(logits.shape) and item['raw_float32_bytes'] == logits.nbytes
            assert item['serialized_bytes'] == path.stat().st_size
            with np.load(path, allow_pickle=False) as archive:
                compact = int(archive['codec']) == 1
                assert item['codec'] == ('exact_high16' if compact else 'raw_float32')
                assert item['payload_bytes'] == logits.nbytes//(2 if compact else 1)
                assert item['shape_and_data_sha256'] == archive['data_sha256'].tobytes().hex()
            actual = independent_losses(logits, ids)
            np.testing.assert_allclose(actual, metric['token_NLL'], rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(actual.mean(), metric['mean_NLL_FP64'], rtol=1e-12, atol=1e-12)
            losses[stage].append({'batch_index': batch, 'token_NLL': actual.tolist()})
            if args.seed != p['seeds'][0] and stage != 'native':
                first_dir = root/'holdout_native_v2'/str(p['seeds'][0])
                first_audit = json.loads((first_dir/'independent_audit.json').read_text())
                assert first_audit['status'] == 'full_reserved_case_raw_losses_controls_coverage_and_resources_independently_verified'
                assert first_audit['source_sha256'] == sha(Path(__file__)) and first_audit['shared_protocol_sha256'] == sha(args.protocol)
                assert first_audit['results_sha256'] == sha(first_dir/'results.json')
                first_result = json.loads((first_dir/'results.json').read_text())
                first = first_dir/path.name
                first_outputs = [x for x in first_result['outputs'] if x['batch'] == batch and x['stage'] == stage]
                assert len(first_outputs) == 1 and first_outputs[0]['sha256'] == sha(first)
                np.testing.assert_array_equal(logits.view(np.uint32), read_logits(first).view(np.uint32))
    summary = {}
    for stage in STAGES:
        short, long = assemble_losses(plan, losses[stage])
        mean = float((short.sum()+long.sum())/10240)
        summary[stage] = {'mean_NLL_FP64': mean, 'subset_perplexity': float(np.exp(mean)),
            'short_mean_NLL': float(short.mean()), 'long_mean_NLL': float(long.mean()), 'targets': 10240}
    audit = {'status': 'full_reserved_case_raw_losses_controls_coverage_and_resources_independently_verified',
        'source_sha256': sha(Path(__file__)), 'results_sha256': sha(out/'results.json'),
        'shared_protocol_sha256': sha(args.protocol), 'seed': args.seed, 'samples': p['samples'],
        'pbits_per_parameter': 1, 'unique_parameter_coordinates': 1728801536,
        'targets': 10240, 'articles': 40, 'losses': summary, **resources,
        'scope': 'Complete frozen teacher-forced article holdout raw-logit losses and shape/program/receipt audit. Deterministic controls bitwise checked against first seed for later seeds. Same-fit weight hashes bound to independently audited master. Full model native trajectories not replayed; sampler primitives covered by component audits. One case, not a five-seed quality conclusion or numerical BF16/hardware proof.'}
    with (out/'independent_audit.json').open('x') as f:
        json.dump(audit, f, indent=2)
    print(json.dumps(audit))


if __name__ == '__main__':
    main()
