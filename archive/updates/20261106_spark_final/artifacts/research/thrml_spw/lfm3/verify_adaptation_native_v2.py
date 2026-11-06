"""Audit direct-program full coverage, saved native losses and replay fidelity."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda: f.read(16 * 1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=['baseline', 'mean', 'binomial'], required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--replay', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    root = Path('results/thrml_spw/lfm3_1p7b')
    shared = json.loads((root / 'adaptation_native_v2_protocol.json').read_text())
    out = root / 'adaptation_native_v2' / ('initial_replay' if args.replay else f'{args.arm}/{args.seed}')
    protocol = json.loads((out / 'protocol.json').read_text())
    result = json.loads((out / 'results.json').read_text())
    for name, digest in shared['sources_sha256'].items():
        assert sha(Path(__file__).with_name(name)) == digest
    for name, digest in shared['auxiliary_sources_sha256'].items():
        assert sha(Path(name)) == digest
    for name, digest in shared['inputs_sha256'].items():
        assert sha(root / name) == digest
    assert protocol['shared_protocol_sha256'] == sha(root / 'adaptation_native_v2_protocol.json')
    assert result['status'] == 'all_exported_parameter_programs_native_validation_complete'
    assert protocol['arm'] == result['arm'] == args.arm
    assert protocol['seed'] == result['seed'] == args.seed
    assert protocol['initial_replay'] == result['initial_replay'] == args.replay
    package_dir = root / 'adaptation_pilot' / args.arm
    assert sha(package_dir / 'independent_audit.json') == protocol['package_audit_sha256']
    assert sha(package_dir / 'program_package.json') == protocol['package_sha256'] == result['package_sha256']
    assert sha(root / 'program_export_gpu_preflight/independent_audit.json') == protocol['GPU_preflight_audit_sha256']
    assert result['worker_ready']['ready'] and any('cuda' in v.lower() or 'gpu' in v.lower() for v in result['worker_ready']['devices'])
    package = json.loads((package_dir / 'program_package.json').read_text())
    records = {v['name']: v for v in package['programs']}
    meta = {v['name']: v for v in json.loads((root / 'meta_inventory.json').read_text())['parameters']}
    aliases = json.loads((package_dir / 'module_aliases.json').read_text())
    assert set(records) == set(meta) and len(records) == result['unique_programs'] == 286
    assert len(aliases) == result['weighted_modules'] == 287
    assert set(aliases) == {v.removesuffix('.weight') for v in meta} | {'lm_head'}
    for module, canonical in aliases.items():
        assert canonical == ('model.embed_tokens.weight' if module == 'lm_head' else module + '.weight')
    for name, rec in records.items():
        assert rec['shape'] == meta[name]['shape'] and rec['numel'] == meta[name]['numel']
        assert sha(Path(rec['program'])) == rec['program_sha256']
        with np.load(rec['program']) as program:
            field = program['codes'].astype(np.float32) * np.float32(8 / 4095) - np.float32(4)
            probability = 1 / (1 + np.exp(-2 * field.astype(np.float64)))
            rows = field.shape[0] if field.ndim > 1 else 1
            mean = (program['lower'].astype(np.float64) + program['span'].astype(np.float64)
                    * probability.reshape(rows, -1, int(program['group_size']))).reshape(field.shape)
            weight = torch.from_numpy(mean).to(torch.bfloat16)
            assert hashlib.sha256(weight.view(torch.uint16).numpy().tobytes()).hexdigest() == result['same_fit_BF16_weight_hashes'][name]
    assert set(result['same_fit_BF16_weight_hashes']) == set(records)
    assert sum(v['numel'] for v in meta.values()) == result['unique_parameter_coordinates'] == 1728801536
    if args.replay:
        assert args.arm == 'baseline' and args.seed == 202610300
        ids_all = np.load(root / 'language_subset/tokens.npz')['tokens']
    else:
        assert args.seed in shared['seeds']
        ids_all = np.load(root / 'adaptation_data/tokens.npz')['validation_tokens']
    batches = len(ids_all) // 8
    assert len(result['outputs']) == batches and result['targets'] == len(ids_all) * 64
    assert len(result['metrics']) == batches * 4
    losses = {stage: [] for stage in ['BF16', 'same_fit_BF16', 'nominal_controls', 'native']}
    replay_compared = []
    for batch, output in enumerate(result['outputs']):
        path = Path(output['path'])
        assert sha(path) == output['sha256']
        with np.load(path) as arrays:
            ids = arrays['input_ids']
            np.testing.assert_array_equal(ids, ids_all[batch * 8:(batch + 1) * 8])
            baseline = arrays['BF16']
            for stage in losses:
                y = arrays[stage]
                assert y.dtype == np.float32 and y.shape == (8, 65, 128000) and np.isfinite(y).all()
                rec = next(v for v in result['metrics'] if v['batch'] == batch and v['stage'] == stage)
                nll = []
                for window in range(8):
                    values = F.cross_entropy(torch.from_numpy(y[window, :-1].astype(np.float64)),
                        torch.from_numpy(ids[window, 1:]), reduction='none').numpy()
                    np.testing.assert_allclose(values, rec['token_NLL'][window], rtol=1e-12, atol=1e-12)
                    nll.extend(values.tolist())
                mean = float(np.mean(nll))
                np.testing.assert_allclose(mean, rec['mean_NLL_FP64'], rtol=1e-12, atol=1e-12)
                np.testing.assert_allclose(np.exp(mean), rec['subset_perplexity'], rtol=1e-12, atol=1e-12)
                np.testing.assert_allclose(np.mean((y.astype(np.float64) - baseline) ** 2), rec['logit_MSE_vs_BF16'], rtol=1e-12, atol=1e-12)
                losses[stage].extend(nll)
                if args.replay and stage in ['BF16', 'nominal_controls', 'native']:
                    reference = root / 'language_native_v2/202610300/outputs.npz'
                    with np.load(reference) as prior:
                        np.testing.assert_array_equal(y, prior[stage])
                    replay_compared.append(stage)
            if not args.replay:
                reference = root / 'adaptation_native_v2/baseline' / str(shared['seeds'][0]) / f'batch{batch}.npz'
                if path != reference:
                    with np.load(reference) as prior:
                        np.testing.assert_array_equal(baseline, prior['BF16'])
                reference = root / 'adaptation_native_v2' / args.arm / str(shared['seeds'][0]) / f'batch{batch}.npz'
                if path != reference:
                    with np.load(reference) as prior:
                        np.testing.assert_array_equal(arrays['nominal_controls'], prior['nominal_controls'])
                        np.testing.assert_array_equal(arrays['same_fit_BF16'], prior['same_fit_BF16'])
    order = None
    retained = warmup = coordinates = 0
    for batch in range(batches):
        for stage in ['nominal_controls', 'native']:
            calls = [v for v in result['operator_calls'] if v['batch'] == batch and v['stage'] == stage]
            assert len(calls) == 287 and Counter(v['module'] for v in calls) == Counter(aliases.keys())
            if order is None:
                order = [v['module'] for v in calls]
            assert [v['module'] for v in calls] == order
            for index, rec in enumerate(calls):
                canonical = aliases[rec['module']]
                assert rec['canonical_program'] == canonical
                assert rec['program_sha256'] == records[canonical]['program_sha256']
                assert rec['call'] == 287 + batch * 287 + index
                shape = meta[canonical]['shape']
                kind = ('embedding' if rec['module'] == 'model.embed_tokens' else 'linear' if len(shape) == 2
                        else 'conv' if len(shape) == 3 else 'elementwise')
                assert rec['kind'] == kind and rec['ok'] and rec['finite']
                assert rec['program_codes_used_directly'] and not rec['reference_weights_supplied']
                assert rec['range_metadata_dtype'] == 'float32' and rec['field_bits'] == rec['activation_bits'] == 12
                if kind == 'embedding':
                    rows = np.unique(ids_all[batch * 8:(batch + 1) * 8]).tolist()
                    assert rec['embedding_rows'] == rows
                    fanin, channels = 1, len(rows) * shape[1]
                elif kind == 'linear':
                    fanin, channels = shape[1], shape[0]
                elif kind == 'conv':
                    fanin, channels = 7, shape[0]
                else:
                    fanin, channels = 1, shape[0]
                n = fanin * channels
                padding = fanin * (((channels + 63) // 64) * 64 - channels)
                assert rec['weight_coordinates'] == n and rec['padding_nodes'] == padding
                assert rec['retained_evaluations'] == ((n + padding) * shared['samples'] if stage == 'native' else 0)
                assert rec['warmup_evaluations'] == (n + padding if stage == 'native' else 0)
                if stage == 'native':
                    coordinates += n; retained += rec['retained_evaluations']; warmup += rec['warmup_evaluations']
    assert len(result['operator_calls']) == 2 * batches * 287
    if args.replay:
        prior = json.loads((root / 'language_native_v2/202610300/results.json').read_text())
        for stage in ['nominal_controls', 'native']:
            assert [v['module'] for v in prior['operator_calls'] if v['stage'] == stage] == order
        assert replay_compared == ['BF16', 'nominal_controls', 'native']
    summary = {stage: {'mean_NLL_FP64': float(np.mean(values)), 'subset_perplexity': float(np.exp(np.mean(values)))} for stage, values in losses.items()}
    audit = {'status': 'all_direct_program_coverage_losses_and_resource_receipts_verified',
        'source_sha256': sha(Path(__file__)), 'results_sha256': sha(out / 'results.json'),
        'arm': args.arm, 'seed': args.seed, 'initial_replay_exact_stages': replay_compared,
        'unique_parameter_coordinates': 1728801536, 'targets': len(ids_all) * 64,
        'losses': summary, 'retained_scheduled_evaluations': retained, 'warmup_scheduled_evaluations': warmup,
        'sampled_coordinates_across_uses': coordinates,
        'scope': 'All canonical finite programs and hashes, tied aliases, same-fit BF16 control weight hashes, complete operator coverage, independent saved-logit losses and scheduled-work bookkeeping. Initial replay additionally requires bitwise whole-model equality to the original unadapted run. Native full trajectories not replayed at model scale; direct worker public trajectories validated by GPU preflight. Conditional validation subset only, no generalization, equivalence or physical-performance proof.'}
    with (out / 'independent_audit.json').open('x') as f:
        json.dump(audit, f, indent=2)
    print(json.dumps(audit))


if __name__ == '__main__':
    main()
