"""Explicit logical payload and scheduled-work accounting, not hardware estimates."""
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path('results/thrml_spw/lfm3_1p7b')
    out = root / 'language_resource_model'
    out.mkdir(exist_ok=True)
    meta_path = root / 'meta_inventory.json'
    tokens_path = root / 'language_subset/tokens.npz'
    protocol_path = root / 'language_native_v2_protocol.json'
    meta = json.loads(meta_path.read_text())
    protocol = json.loads(protocol_path.read_text())
    tokens = np.load(tokens_path)['tokens']
    n = sum(p['numel'] for p in meta['parameters'])
    assert n == 1728801536
    groups = 0
    for p in meta['parameters']:
        fanin = int(np.prod(p['shape'][1:])) if len(p['shape']) > 1 else p['numel']
        size = min(128, fanin)
        assert fanin % size == 0
        groups += p['numel'] // size
    assert groups == 13531430
    metadata_bytes = groups * 8  # Two FP32 range values, assumed packed digital payload.
    payloads = []
    for control_bits in [4, 6, 8, 10, 12, 16, 32]:
        packed = (n * (1 + control_bits) + 7) // 8 + metadata_bytes
        payloads.append({'control_bits_per_parameter': control_bits,
                         'packed_binary_state_bytes': (n + 7) // 8,
                         'packed_commands_bytes': (n * control_bits + 7) // 8,
                         'range_metadata_bytes': metadata_bytes,
                         'combined_packed_bytes': packed,
                         'combined_bits_per_parameter': packed * 8 / n,
                         'combined_fraction_of_BF16_weight_bytes': packed / (2 * n)})
    conventional = [{'weight_bits': b, 'packed_weights_plus_FP32_ranges_bytes':
                     (n * b + 7) // 8 + metadata_bytes} for b in [2, 4, 8]]
    uses = []
    fanins = set()
    for p in meta['parameters']:
        shape = p['shape']
        if p['name'] == 'model.embed_tokens.weight':
            fanin, channels, kind = 1, len(np.unique(tokens)) * shape[1], 'embedding'
        elif len(shape) == 1:
            fanin, channels, kind = 1, shape[0], 'elementwise'
        else:
            fanin, channels = int(np.prod(shape[1:])), shape[0]
            kind = 'conv' if len(shape) == 3 else 'linear'
        pad = fanin * ((channels + 63) // 64 * 64 - channels)
        uses.append({'name': p['name'].removesuffix('.weight'), 'kind': kind,
                     'used_coordinates': fanin * channels, 'padding_nodes': pad})
        fanins.add(fanin)
    embedding = next(p for p in meta['parameters'] if p['name'] == 'model.embed_tokens.weight')
    uses.append({'name': 'lm_head', 'kind': 'linear', 'used_coordinates': embedding['numel'],
                 'padding_nodes': 0})
    fanins.add(embedding['shape'][1])
    slots = sum(p['used_coordinates'] + p['padding_nodes'] for p in uses)
    samples = protocol['samples']
    comparisons = []
    for k in [2, 4, 8]:
        assert samples % k == 0
        comparisons.append({'spatial_pbits_per_parameter': k,
                            'spatial_logical_pbits_if_fully_resident': k * n,
                            'single_samples': samples,
                            'spatial_samples_for_equal_unpadded_retained_node_evaluations': samples // k,
                            'both_unpadded_parameter_level_retained_evaluations': n * samples,
                            'single_over_spatial_sampling_latency_if_same_rate_and_full_parallelism': k,
                            'single_over_spatial_transition_energy_if_same_energy_per_evaluation': 1,
                            'single_over_spatial_programming_energy_if_same_cost_per_node': 1 / k})
    result = {
        'source_sha256': sha(Path(__file__)),
        'inputs_sha256': {str(p): sha(p) for p in [meta_path, tokens_path, protocol_path]},
        'unique_parameters': n, 'range_groups': groups, 'range_metadata_FP32_pair_bytes': metadata_bytes,
        'BF16_reference_weight_bytes': 2 * n, 'logical_single_pbits_if_fully_resident': n,
        'single_payload_sweep': payloads, 'conventional_quantized_payload_assumptions': conventional,
        'unique_embedding_rows_used': len(np.unique(tokens)), 'operator_uses': uses,
        'sampled_coordinates_across_operator_uses': sum(p['used_coordinates'] for p in uses),
        'padding_nodes_across_operator_uses': sum(p['padding_nodes'] for p in uses),
        'per_seed_retained_node_evaluations_including_padding': slots * samples,
        'per_seed_warmup_node_evaluations_including_padding': slots,
        'five_seed_retained_node_evaluations_including_padding': slots * samples * len(protocol['seeds']),
        'five_seed_warmup_node_evaluations_including_padding': slots * len(protocol['seeds']),
        'max_cached_bank_nodes': max(fanins) * 64,
        'sum_cached_bank_nodes_across_unique_fanin_shapes': sum(fanins) * 64,
        'largest_linear_int32_accumulator_bytes_for_520_positions': 4 * tokens.size * (max(fanins) // 128) * 64,
        'matched_spatial_temporal_identities': comparisons,
        'scope': 'Analytical bookkeeping for current software mapping; full-model node-work predictions await comparison against completed audited native receipts. Logical fully-resident p-bit count differs from bounded simulator banks. Single payloads assume packed B-bit controls, one packed binary state per weight and two digital FP32 range values per group. Analog control area is not measured or converted to bytes. Current uint16 controls are16 storage bits, not packed12. Original BF16 checkpoint, framework/node/program indices, actual unpacked states and FP32 fields, activations, compiler workspace, accumulators and IO are additional. INT2/4/8 are simulated weight-only quality references, not matched transition resources. Spatial comparisons are algebraic identities conditional on independently programmable nodes, same sampling rate and per-node energy, full parallelism and matched quality; no empirical K-p-bit model quality, physical advantage or absolute energy/latency estimate.'}
    with (out / 'results.json').open('x') as f:
        json.dump(result, f, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k not in ['operator_uses', 'scope']}))


if __name__ == '__main__':
    main()
