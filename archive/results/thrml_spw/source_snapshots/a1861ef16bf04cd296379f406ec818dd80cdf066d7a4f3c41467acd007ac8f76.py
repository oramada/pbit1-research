"""Device-capacity arithmetic and conditional sample-round times, not benchmarks."""
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path('results/thrml_spw/lfm3_1p7b'); out = root / 'published_hardware_context'
    paths = [out / 'claims.json', root / 'meta_inventory.json', root / 'layer_allocation_candidates/results.json',
             root / 'language_resource_model/results.json']
    claims, meta, policy, payload = [json.loads(p.read_text()) for p in paths]
    n = sum(p['numel'] for p in meta['parameters']); assert n == 1728801536
    chip = claims['chip_pbits_figure_value']; card = chip * claims['chips_per_announced_card']
    capacities = []
    for k in [1, 2, 4, 8]:
        nodes = k * n
        capacities.append({'pbits_per_parameter': k, 'logical_weight_devices': nodes,
            'capacity_only_chip_count': (nodes + chip - 1) // chip,
            'capacity_only_card_count': (nodes + card - 1) // card})
    cycles = {'uniform': 287 * 1024, 'allocated': sum(policy['selected_samples'].values())}
    scenarios = [{'assumed_independent_observations_per_second': f,
                  'serial_sample_round_time_seconds': {k: v / f for k, v in cycles.items()}}
                 for f in [1000000, 10000000, 50000000]]
    result = {'status': 'conditional_capacity_and_serial_round_scenarios_only', 'source_sha256': sha(Path(__file__)),
        'inputs_sha256': {str(p): sha(p) for p in paths}, 'unique_parameters': n,
        'company_figure_chip_pbits': chip, 'derived_nominal_card_pbits': card, 'capacities': capacities,
        'serial_weighted_operator_sample_cycles': cycles, 'conditional_frequency_scenarios': scenarios,
        'one_pbit_state_control_range_payload_bytes': next(v['combined_packed_bytes'] for v in payload['single_payload_sweep'] if v['control_bits_per_parameter'] == 12),
        'BF16_reference_weight_bytes': payload['BF16_reference_weight_bytes'],
        'measured_hardware_energy_per_inference_J': None, 'measured_hardware_latency_seconds': None,
        'scope': 'Capacity-only rounding assuming every announcednodeusableforoneindependentprogrammableweight; not feasiblelayout/controller/DAC/readout/metadata/networkmapping proof. K2/K4/K8 are internalrepresentationcontrols, notassertedvendorformats. Frequencies are hypotheticalINDEPENDENTobservationrates;50MHz numericalreference is notvalidatedindependence fromannouncedsamplerrate. SerializedsumT/f assumesalloperatorweightsparallel,nooverlap,zeroarithmetic/program/readoutcost; nothardwarelatency/lowerbound orDAGschedule. Powerclaim is not converted into transitionenergy; physicalsamples,updatesandflipsdiffer.'}
    with (out / 'results.json').open('x') as f: json.dump(result, f, indent=2)
    print(json.dumps({'capacity_only_K1': capacities[0], 'cycles': cycles}))


if __name__ == '__main__':
    main()
