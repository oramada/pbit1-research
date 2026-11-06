"""Verify arithmetic/labels, not the vendor's physical performance claims."""
from decimal import Decimal
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path('results/thrml_spw/lfm3_1p7b'); out = root / 'published_hardware_context'
    r = json.loads((out / 'results.json').read_text()); claims = json.loads((out / 'claims.json').read_text())
    assert r['status'] == 'conditional_capacity_and_serial_round_scenarios_only'
    assert r['source_sha256'] == sha(Path(__file__).with_name('published_hardware_context.py'))
    for name, digest in r['inputs_sha256'].items(): assert sha(Path(name)) == digest
    assert claims['classification'] == 'company_published_design_and_roadmap_claims_not_project_measurements'
    n = sum(v['numel'] for v in json.loads((root / 'meta_inventory.json').read_text())['parameters'])
    chip, card = claims['chip_pbits_figure_value'], claims['chip_pbits_figure_value'] * claims['chips_per_announced_card']
    assert card == r['derived_nominal_card_pbits']
    for rec in r['capacities']:
        nodes = rec['pbits_per_parameter'] * n
        assert nodes == rec['logical_weight_devices']
        for field, size in [('capacity_only_chip_count', chip), ('capacity_only_card_count', card)]:
            count = rec[field]; assert (count - 1) * size < nodes <= count * size
    policy = json.loads((root / 'layer_allocation_candidates/results.json').read_text())['selected_samples']
    assert r['serial_weighted_operator_sample_cycles'] == {'uniform': 293888, 'allocated': sum(policy.values())}
    for rec in r['conditional_frequency_scenarios']:
        f = Decimal(rec['assumed_independent_observations_per_second'])
        for kind, cycles in r['serial_weighted_operator_sample_cycles'].items():
            assert abs(Decimal(str(rec['serial_sample_round_time_seconds'][kind])) * f - cycles) < Decimal('1e-8')
    payload = json.loads((root / 'language_resource_model/results.json').read_text())
    assert r['one_pbit_state_control_range_payload_bytes'] == next(v['combined_packed_bytes'] for v in payload['single_payload_sweep'] if v['control_bits_per_parameter'] == 12)
    assert r['measured_hardware_energy_per_inference_J'] is None and r['measured_hardware_latency_seconds'] is None
    assert claims['independent_sample_rate_measured_for_this_work'] is None
    audit = {'status': 'conditional_arithmetic_and_measurement_separation_verified',
        'source_sha256': sha(Path(__file__)), 'results_sha256': sha(out / 'results.json'),
        'scope': 'Arithmetic/hash/claim-vs-assumption labels only. Does not verify physicalvendorclaims, feasibility, actualavailability, clockindependence, energy/latencyorapplicationquality.'}
    with (out / 'independent_audit.json').open('x') as f: json.dump(audit, f, indent=2)
    print(json.dumps(audit))


if __name__ == '__main__':
    main()
