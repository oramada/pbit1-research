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
    parser.add_argument('--budget', type=int, choices=[1024, 4096, 8192], required=True)
    parser.add_argument('--word-bits', type=int, choices=[4, 8], required=True)
    args = parser.parse_args()
    assert args.arm == 'mean' and not args.replay
    torch.set_num_threads(2)
    root = Path('results/thrml_spw/lfm3_1p7b')
    amendment = json.loads((root / 'h100_backend_control_amendment_v3.json').read_text())
    assert amendment['new_auditor_sha256'] == sha(Path(__file__))
    assert amendment['original_producer_protocol_sha256'] == sha(root / 'modal_h100_word_protocol_v2.json')
    assert amendment['H100_reference_sha256'] == sha(root / 'h100_reference_controls_v3.json')
    shared = json.loads((root / 'modal_h100_word_protocol_v2.json').read_text())
    shared['seeds'] = shared['seeds_by_budget'][str(args.budget)]
    shared['samples'] = args.budget // args.word_bits
    out = root / 'modal_h100_word_v2' / f'C{args.budget}' / f'K{args.word_bits}' / args.arm / str(args.seed)
    word_dir = root / 'adapted_word_package'
    word_package = json.loads((word_dir / 'package.json').read_text())
    word_records = {v['name']: v for v in word_package['programs']}
    protocol = json.loads((out / 'protocol.json').read_text())
    result = json.loads((out / 'results.json').read_text())
    for name, digest in shared['sources_sha256'].items():
        assert sha(Path(__file__).with_name(name)) == digest
    for name, digest in shared['auxiliary_sources_sha256'].items():
        assert sha(Path(name)) == digest
    for name, digest in shared['inputs_sha256'].items():
        assert sha(root / name) == digest
    assert protocol['shared_protocol_sha256'] == sha(root / 'modal_h100_word_protocol_v2.json')
    assert result['status'] == 'all_adapted_word_parameters_native_validation_complete'
    assert protocol['arm'] == result['arm'] == args.arm
    assert protocol['seed'] == result['seed'] == args.seed
    assert protocol['initial_replay'] == result['initial_replay'] == args.replay
    assert protocol['word_bits'] == result['word_bits'] == args.word_bits
    assert protocol['budget'] == result['budget'] == args.budget
    assert protocol['samples'] == result['samples'] == shared['samples']
    assert protocol['word_package_sha256'] == result['word_package_sha256'] == sha(word_dir / 'package.json')
    assert protocol['word_package_audit_sha256'] == sha(word_dir / 'independent_audit.json')
    assert protocol['word_GPU_preflight_audit_sha256'] == sha(root / 'adapted_word_worker_gpu_preflight/independent_audit.json')
    word_audit = json.loads((word_dir / 'independent_audit.json').read_text())
    assert word_audit['status'] == 'every_K4_K8_word_and_original_metadata_byte_verified'
    assert word_audit['package_sha256'] == sha(word_dir / 'package.json')
    package_dir = root / 'adaptation_pilot' / args.arm
    assert sha(package_dir / 'independent_audit.json') == protocol['package_audit_sha256']
    assert sha(package_dir / 'program_package.json') == protocol['package_sha256'] == result['package_sha256']
    assert sha(root / 'program_export_gpu_preflight/independent_audit.json') == protocol['GPU_preflight_audit_sha256']
    assert result['worker_ready']['ready'] and any('cuda' in v.lower() or 'gpu' in v.lower() for v in result['worker_ready']['devices'])
    package = json.loads((package_dir / 'program_package.json').read_text())
    records = {v['name']: v for v in package['programs']}
    meta = {v['name']: v for v in json.loads((root / 'meta_inventory.json').read_text())['parameters']}
    aliases = json.loads((package_dir / 'module_aliases.json').read_text())
    assert set(word_records) == set(records)
    assert set(records) == set(meta) and len(records) == result['unique_programs'] == 286
    assert len(aliases) == result['weighted_modules'] == 287
    assert set(aliases) == {v.removesuffix('.weight') for v in meta} | {'lm_head'}
    for module, canonical in aliases.items():
        assert canonical == ('model.embed_tokens.weight' if module == 'lm_head' else module + '.weight')
    for name, rec in records.items():
        assert rec['shape'] == meta[name]['shape'] and rec['numel'] == meta[name]['numel']
        assert sha(Path(rec['program'])) == rec['program_sha256']
        with np.load(rec['program']) as program:
            wr = word_records[name]
            item = wr['outputs'][str(args.word_bits)]
            assert wr['original_program_sha256'] == rec['program_sha256']
            assert sha(Path(item['path'])) == item['sha256']
            with np.load(item['path']) as word:
                assert word['words'].dtype == np.uint8 and list(word['words'].shape) == rec['shape']
                assert int(word['bits']) == args.word_bits and int(word['format_version']) == 2 and int(word['sign_field_bound']) == 4
                for key in ['lower', 'span', 'shape', 'group_size']:
                    a, b = word[key], program[key]
                    assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
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
            reference = json.loads((root / 'h100_reference_controls_v3.json').read_text())
            for stage in ['BF16', 'same_fit_BF16']:
                assert hashlib.sha256(arrays[stage].tobytes()).hexdigest() == reference['array_sha256'][f'{batch}/{stage}']
            first_case = root / 'modal_h100_word_v2' / f'C{args.budget}' / f'K{args.word_bits}' / args.arm / str(shared['seeds'][0]) / f'batch{batch}.npz'
            if path != first_case:
                with np.load(first_case) as prior:
                    np.testing.assert_array_equal(arrays['nominal_controls'].view(np.uint32), prior['nominal_controls'].view(np.uint32))
    reference_dir = root / {1024: 'counted_mean_validation', 4096: 'counted_mean_t4096', 8192: 'counted_mean_t8192'}[args.budget] / 'mean' / str(args.seed)
    reference_audit = json.loads((reference_dir / 'independent_audit.json').read_text())
    reference_result = json.loads((reference_dir / 'results.json').read_text())
    assert reference_audit['status'] == 'all_direct_program_coverage_losses_and_resource_receipts_verified'
    assert reference_audit['source_sha256'] == shared['reference_verifier_sha256'][str(args.budget)]
    assert reference_audit['results_sha256'] == sha(reference_dir / 'results.json')
    assert reference_audit['seed'] == args.seed and reference_audit['targets'] == 2048
    reference_calls = {(v['batch'], v['stage'], v['module']): v for v in reference_result['operator_calls']}
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
                assert rec['source_single_program_sha256'] == records[canonical]['program_sha256']
                assert rec['program_sha256'] == word_records[canonical]['outputs'][str(args.word_bits)]['sha256']
                assert rec['call'] == 287 + batch * 287 + index
                shape = meta[canonical]['shape']
                kind = ('embedding' if rec['module'] == 'model.embed_tokens' else 'linear' if len(shape) == 2
                        else 'conv' if len(shape) == 3 else 'elementwise')
                assert rec['kind'] == kind and rec['ok'] and rec['finite']
                assert rec['program_words_used_directly'] and not rec['reference_weights_supplied']
                assert rec['range_metadata_dtype'] == 'float32' and rec['activation_bits'] == 12
                assert rec['word_bits'] == rec['pbits_per_parameter'] == args.word_bits
                assert rec['programmable_field_levels_per_node'] == 2 and rec['field_bound'] == 4
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
                tiles = (channels + 63) // 64
                assert rec['tile_lanes_limit'] == 4
                assert rec['max_simultaneous_sampler_state_slots'] == min(4, tiles) * fanin * 64
                assert rec['simulation_observer'] == 'per_weight_counts_per_bitplane_then_integer_contraction'
                assert rec['simulation_count_carry_bytes'] == (4 * min(4, tiles) * fanin * 64 if stage == 'native' else 0)
                assert rec['weight_coordinates'] == n and rec['padding_parameter_slots'] == padding
                assert rec['padding_nodes'] == args.word_bits * padding
                assert rec['retained_evaluations'] == (args.word_bits * (n + padding) * shared['samples'] if stage == 'native' else 0)
                assert rec['warmup_evaluations'] == (args.word_bits * (n + padding) if stage == 'native' else 0)
                paired = reference_calls[(batch, stage, rec['module'])]
                assert paired['kind'] == kind and paired['weight_coordinates'] == n and paired['padding_nodes'] == padding
                assert rec['retained_evaluations'] == paired['retained_evaluations']
                assert rec['warmup_evaluations'] == args.word_bits * paired['warmup_evaluations']
                if stage == 'native':
                    coordinates += n; retained += rec['retained_evaluations']; warmup += rec['warmup_evaluations']
    assert len(result['operator_calls']) == 2 * batches * 287
    if args.replay:
        prior = json.loads((root / 'language_native_v2/202610300/results.json').read_text())
        for stage in ['nominal_controls', 'native']:
            assert [v['module'] for v in prior['operator_calls'] if v['stage'] == stage] == order
        assert replay_compared == ['BF16', 'nominal_controls', 'native']
    assert retained == reference_audit['retained_scheduled_evaluations']
    assert warmup == args.word_bits * reference_audit['warmup_scheduled_evaluations']
    assert coordinates == reference_audit['sampled_coordinates_across_uses']
    summary = {stage: {'mean_NLL_FP64': float(np.mean(values)), 'subset_perplexity': float(np.exp(np.mean(values)))} for stage, values in losses.items()}
    audit = {'status': 'all_adapted_word_coverage_losses_controls_and_matched_work_verified',
        'source_sha256': sha(Path(__file__)), 'results_sha256': sha(out / 'results.json'),
        'arm': args.arm, 'seed': args.seed, 'word_bits': args.word_bits, 'budget': args.budget,
        'samples': shared['samples'], 'reference_single_audit_sha256': sha(reference_dir / 'independent_audit.json'),
        'all_H100_BF16_controls_bitwise_equal_to_H100_master': True,
        'local_BF16_control_bitwise_equality_required': False,
        'backend_control_amendment_sha256': sha(root / 'h100_backend_control_amendment_v3.json'),
        'retained_work_exactly_matches_same_budget_single': True,
        'initial_replay_exact_stages': replay_compared,
        'unique_parameter_coordinates': 1728801536, 'targets': len(ids_all) * 64,
        'losses': summary, 'retained_scheduled_evaluations': retained, 'warmup_scheduled_evaluations': warmup,
        'sampled_coordinates_across_uses': coordinates,
        'max_simultaneous_sampler_state_slots': max(v['max_simultaneous_sampler_state_slots'] for v in result['operator_calls']),
        'max_temporary_weight_count_bytes': max(v['simulation_count_carry_bytes'] for v in result['operator_calls']),
        'simulation_observer': 'per_weight_counts_per_bitplane_then_integer_contraction',
        'scope': 'All original adapted finite programs and calibrated spatial words/hashes, unchanged range bytes/tied aliases, original same-fit BF16 weight hashes, all saved losses and exact BF16/same-fit control bits versus the H100 master; local control bitwise equality is not required under the explicitly amended backend policy. Every operator retained work matches paired same-budget single, warmup K times separately charged. Observer carry excludes total runtime/replay memory. Full model trajectories not replayed; GPU component public states validated. Conditional validation only; no generalization, numerical equivalence or physical-performance proof.'}
    with (out / 'h100_independent_audit_v3.json').open('x') as f:
        json.dump(audit, f, indent=2)
    print(json.dumps(audit))


if __name__ == '__main__':
    main()
