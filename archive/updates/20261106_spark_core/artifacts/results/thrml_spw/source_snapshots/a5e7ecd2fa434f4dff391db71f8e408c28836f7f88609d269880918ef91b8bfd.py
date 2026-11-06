"""Offline exact state/counter/readout/resource audit for the batched worker."""
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path('results/thrml_spw/lfm3_1p7b'); out = root / 'batched_worker_preflight'
    p = json.loads((out / 'protocol.json').read_text()); r = json.loads((out / 'results.json').read_text())
    assert r['status'] == 'all38_batched_worker_outputs_match_reference_producer' and len(r['records']) == 38
    assert sha(out / 'protocol.json') == r['protocol_sha256']
    for name, digest in p['sources_sha256'].items(): assert sha(Path(__file__).with_name(name)) == digest
    for name, digest in r['artifacts_sha256'].items(): assert sha(Path(name)) == digest
    original = root / 'program_export_gpu_preflight'
    assert sha(original / 'results.json') == p['original_GPU_results_sha256']
    assert sha(original / 'independent_audit.json') == p['original_GPU_audit_sha256']
    assert sha(root / 'batched_bank_preflight/independent_audit.json') == p['bank_audit_sha256']
    original_files = json.loads((original / 'results.json').read_text())['artifacts_sha256']
    bits = integers = 0
    for rec in r['records']:
        req, receipt = rec['request'], rec['receipt']
        reference = Path(rec['reference_output'])
        if str(reference) in original_files: assert sha(reference) == original_files[str(reference)]
        np.testing.assert_array_equal(np.load(req['output']), np.load(reference))
        with np.load(req['input']) as a:
            fanin, channels = a['codes'].shape; size = min(128, fanin); groups = fanin // size
            tiles = (channels + 63) // 64; padding = fanin * (tiles * 64 - channels)
            assert receipt['weight_coordinates'] == fanin * channels and receipt['padding_nodes'] == padding
            assert receipt['max_simultaneous_sampler_state_slots'] == min(4, tiles) * fanin * 64
            assert receipt['tile_batches'] == (tiles + 3) // 4
            native = req['mode'] == 'native'; samples = req['samples']
            assert receipt['retained_evaluations'] == ((fanin * channels + padding) * samples if native else 0)
            assert receipt['warmup_evaluations'] == (fanin * channels + padding if native else 0)
            if not native: continue
            x = a['x']; scale = float(np.abs(x).max()) / 2047 if np.any(x) else 1.
            xq = np.rint(x / scale).clip(-2047, 2047).astype(np.int64)
            expected_outputs = []
            for tile in range(tiles):
                filename = f'call{req["call"]}_tile{tile}.npz'
                refstate = Path(rec['reference_states']) / filename
                if str(refstate) in original_files: assert sha(refstate) == original_files[str(refstate)]
                with np.load(out / 'new_states' / filename) as new, np.load(refstate) as old:
                    for key in ['states', 'codes', 'total', 'inputs']: np.testing.assert_array_equal(new[key], old[key])
                    first = tile * 64; width = min(64, channels - first)
                    states = new['states']; counts = states.sum(0, dtype=np.int64)
                    xt = xq if req['kind'] == 'linear' else np.pad(xq[:, :, first:first + width], ((0, 0), (0, 0), (0, 64 - width)))
                    np.testing.assert_array_equal(new['inputs'], xt)
                    xg = xt.reshape(xt.shape[0], groups, size) if req['kind'] == 'linear' else xt.reshape(xt.shape[0], groups, size, 64)
                    total = (np.stack([xg[:, g] @ counts[g] for g in range(groups)], axis=1) if req['kind'] == 'linear' else (xg * counts[None]).sum(2, dtype=np.int64))
                    np.testing.assert_array_equal(new['total'].astype(np.int64), total)
                    lower = np.pad(a['lower'][:, first:first + width].astype(np.float64), ((0, 0), (0, 64 - width)))
                    span = np.pad(a['span'][:, first:first + width].astype(np.float64), ((0, 0), (0, 64 - width)))
                    offset = xg.sum(2)[:, :, None] if req['kind'] == 'linear' else xg.sum(2)
                    y = np.sum(total.astype(np.float64) / samples * span[None] + offset * lower[None], axis=1) * scale
                    expected_outputs.append(y[:, :width]); bits += states.size; integers += total.size
            np.testing.assert_array_equal(np.load(req['output']), np.concatenate(expected_outputs, axis=1))
    audit = {'status': 'all38_outputs_public_states_counters_readouts_and_resource_receipts_verified',
        'source_sha256': sha(Path(__file__)), 'results_sha256': sha(out / 'results.json'),
        'public_bits_each_serial_and_batched_path': bits, 'integer_entries_each_path': integers,
        'scope': 'Worker/operator fidelity on34prioracceptedcases+4wide/tail cases, exactoriginalpublicstates/counters/outputsandindependentint64/F64readoutreconstruction. No fullmodel parity, timing, quality, physicaladvantage or adoption in existingstudies.'}
    with (out / 'independent_audit.json').open('x') as f: json.dump(audit, f, indent=2)
    print(json.dumps(audit))


if __name__ == '__main__':
    main()
