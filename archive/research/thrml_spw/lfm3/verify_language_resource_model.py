"""Independent closed-form payload/work checks, anchored to prior audited work."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path('results/thrml_spw/lfm3_1p7b')
    out = root / 'language_resource_model'
    r = json.loads((out / 'results.json').read_text())
    assert sha(Path(__file__).with_name('language_resource_model.py')) == r['source_sha256']
    for path, digest in r['inputs_sha256'].items():
        assert sha(Path(path)) == digest
    meta = json.loads((root / 'meta_inventory.json').read_text())
    tokens = np.load(root / 'language_subset/tokens.npz')['tokens']
    old_tokens = np.load(root / 'baseline_smoke.npz')['input_ids']
    old = json.loads((root / 'precision_T1024/independent_audit.json').read_text())
    n = sum(math.prod(p['shape']) for p in meta['parameters'])
    assert n == r['unique_parameters'] == r['logical_single_pbits_if_fully_resident'] == 1728801536
    unique_rows = len(set(int(v) for v in tokens.flat))
    assert unique_rows == r['unique_embedding_rows_used']
    # Tied vocabulary tensor already counted in n; selected embedding lookups
    # are extra uses. Only eleven16-output projections need padded channels.
    padding = sum((64 - p['shape'][0] % 64) % 64 * p['shape'][1]
                  for p in meta['parameters'] if len(p['shape']) == 2 and p['name'] != 'model.embed_tokens.weight')
    assert padding == 11 * 48 * 2048 == r['padding_nodes_across_operator_uses']
    assert r['sampled_coordinates_across_operator_uses'] == n + unique_rows * 2048
    slots = n + unique_rows * 2048 + padding
    difference = (unique_rows - len(set(int(v) for v in old_tokens.flat))) * 2048
    assert slots == old['warmup_evaluations_including_padding'] + difference
    assert r['per_seed_warmup_node_evaluations_including_padding'] == slots
    assert r['per_seed_retained_node_evaluations_including_padding'] == old['retained_evaluations_including_padding'] + difference * 1024 == slots * 1024
    assert r['five_seed_retained_node_evaluations_including_padding'] == slots * 1024 * 5
    assert r['five_seed_warmup_node_evaluations_including_padding'] == slots * 5
    groups = sum(p['numel'] // (7 if len(p['shape']) == 3 else 128) for p in meta['parameters'])
    assert groups == r['range_groups'] == 13531430
    assert r['range_metadata_FP32_pair_bytes'] == 8 * groups
    assert r['BF16_reference_weight_bytes'] == 2 * n
    assert [p['control_bits_per_parameter'] for p in r['single_payload_sweep']] == [4, 6, 8, 10, 12, 16, 32]
    for p in r['single_payload_sweep']:
        bits = p['control_bits_per_parameter']
        assert p['packed_binary_state_bytes'] * 8 == n
        assert p['packed_commands_bytes'] * 8 == bits * n
        assert p['range_metadata_bytes'] == 8 * groups
        payload_bits = n * (bits + 1) + 64 * groups
        assert p['combined_packed_bytes'] * 8 == payload_bits
        assert math.isclose(p['combined_bits_per_parameter'], payload_bits / n, rel_tol=1e-14)
        assert math.isclose(p['combined_fraction_of_BF16_weight_bytes'], payload_bits / (16 * n), rel_tol=1e-14)
    for p in r['conventional_quantized_payload_assumptions']:
        assert p['packed_weights_plus_FP32_ranges_bytes'] * 8 == n * p['weight_bits'] + 64 * groups
    assert len(r['operator_uses']) == 287
    assert len({p['name'] for p in r['operator_uses']}) == 287
    expected_names = {p['name'].removesuffix('.weight') for p in meta['parameters']} | {'lm_head'}
    assert {p['name'] for p in r['operator_uses']} == expected_names
    assert sum(p['used_coordinates'] for p in r['operator_uses']) == n + unique_rows * 2048
    assert sum(p['padding_nodes'] for p in r['operator_uses']) == padding
    assert r['max_cached_bank_nodes'] == 7680 * 64
    assert r['sum_cached_bank_nodes_across_unique_fanin_shapes'] == (1 + 7 + 2048 + 7680) * 64
    assert r['largest_linear_int32_accumulator_bytes_for_520_positions'] == 4 * 520 * 60 * 64
    for p in r['matched_spatial_temporal_identities']:
        k = p['spatial_pbits_per_parameter']
        assert k in [2, 4, 8] and p['spatial_logical_pbits_if_fully_resident'] == k * n
        assert p['single_samples'] == k * p['spatial_samples_for_equal_unpadded_retained_node_evaluations'] == 1024
        assert p['both_unpadded_parameter_level_retained_evaluations'] == n * 1024
        assert p['single_over_spatial_sampling_latency_if_same_rate_and_full_parallelism'] == k
        assert p['single_over_spatial_transition_energy_if_same_energy_per_evaluation'] == 1
        assert p['single_over_spatial_programming_energy_if_same_cost_per_node'] == 1 / k
    audit = {'status': 'all_logical_payload_and_scheduled_work_identities_verified',
             'source_sha256': sha(Path(__file__)), 'results_sha256': sha(out / 'results.json'),
             'prior_T1024_audit_sha256': sha(root / 'precision_T1024/independent_audit.json'),
             'scope': 'Analytical accounting checked against independent dimension formulas and prior audited prompt work plus exact embedding-row difference. Current full-language native receipts still pending. No empirical energy, latency, physical area, storage advantage or matched-quality spatial conclusion.'}
    with (out / 'independent_audit.json').open('x') as f:
        json.dump(audit, f, indent=2)
    print(json.dumps(audit))


if __name__ == '__main__':
    main()
