"""Offline int64 audit of all public samples at 1,028-position worker shapes."""
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    out = Path('results/thrml_spw/lfm3_1p7b/attention_worker_cpu_preflight_v2')
    p = json.loads((out/'protocol.json').read_text())
    r = json.loads((out/'results.json').read_text())
    assert r['status'] == 'all_four_attention_expanded_CPU_worker_cases_produced' and len(r['records']) == 4
    assert sha(out/'protocol.json') == r['protocol_sha256']
    for name, digest in p['sources_sha256'].items():
        assert sha(Path(__file__).with_name(name)) == digest
    for name, digest in p['auxiliary_sources_sha256'].items():
        assert sha(Path(name)) == digest
    for name, digest in r['artifacts_sha256'].items():
        assert sha(Path(name)) == digest
    bits = integers = outputs = 0
    maximum = 0
    for index, rec in enumerate(r['records']):
        req, receipt = rec['request'], rec['receipt']
        with np.load(req['input']) as a:
            fanin, channels = a['codes'].shape
            assert [fanin, channels, req['samples'], req['kind']] == p['cases'][index]
            positions = p['positions_by_case'][index]
            x = a['x']; assert x.shape[0] == positions and x.dtype == np.float32
            scale = float(np.abs(x).max())/2047 if np.any(x) else 1
            assert scale == 1
            xq = np.rint(x/scale).clip(-2047, 2047).astype(np.int64)
            size = min(128, fanin); groups = fanin//size; tiles = (channels+63)//64
            expected_outputs = []
            for tile in range(tiles):
                first = tile*64; width = min(64, channels-first)
                with np.load(out/'states'/f'call{index}_tile{tile}.npz') as raw:
                    z = raw['states']
                    assert z.dtype == np.bool_ and z.shape == (req['samples'], groups, size, 64)
                    counts = z.sum(axis=0, dtype=np.int64)
                    np.testing.assert_array_equal(raw['counts'].astype(np.int64), counts)
                    expected_codes = np.pad(a['codes'][:, first:first+width].reshape(groups, size, width),
                        ((0, 0), (0, 0), (0, 64-width)), constant_values=2048)
                    np.testing.assert_array_equal(raw['codes'], expected_codes)
                    xt = xq if req['kind']=='linear' else np.pad(xq[:, :, first:first+width], ((0, 0), (0, 0), (0, 64-width)))
                    np.testing.assert_array_equal(raw['inputs'], xt)
                    if req['kind']=='linear':
                        xg = xt.reshape(positions, groups, size)
                        total = np.stack([xg[:, g] @ counts[g] for g in range(groups)], axis=1)
                        offset = xg.sum(2)[:, :, None]
                    else:
                        xg = xt.reshape(positions, groups, size, 64)
                        total = (xg*counts[None]).sum(2, dtype=np.int64)
                        offset = xg.sum(2)
                    np.testing.assert_array_equal(raw['total'].astype(np.int64), total)
                    if index == 0:
                        np.testing.assert_array_equal(total[0], -total[1])
                    maximum = max(maximum, int(np.abs(total).max()))
                    assert int(np.abs(total).max()) <= 2047*size*req['samples'] < 2**31
                    lower = np.pad(a['lower'][:, first:first+width].astype(np.float64), ((0, 0), (0, 64-width)))
                    span = np.pad(a['span'][:, first:first+width].astype(np.float64), ((0, 0), (0, 64-width)))
                    y = np.sum(total.astype(np.float64)/req['samples']*span[None] + offset*lower[None], axis=1)*scale
                    expected_outputs.append(y[:, :width]);bits += z.size;integers += total.size
            expected = np.concatenate(expected_outputs, axis=1)
            np.testing.assert_array_equal(np.load(req['output']), expected)
            outputs += expected.size
            nodes = fanin*tiles*64
            assert receipt['retained_evaluations']==nodes*req['samples'] and receipt['warmup_evaluations']==nodes
            assert receipt['simulation_count_workspace_bytes']==4*min(4,tiles)*fanin*64
            assert receipt['max_simultaneous_sampler_state_slots']==min(4,tiles)*fanin*64
            assert receipt['program_codes_used_directly'] and not receipt['reference_weights_supplied']
    result = {'status': 'all_four_attention_expanded_public_states_int64_counters_F64_readouts_and_receipts_verified',
        'source_sha256': sha(Path(__file__)), 'results_sha256': sha(out/'results.json'),
        'public_bits_verified': bits, 'integer_entries_verified': integers,
        'float64_outputs_verified': outputs, 'max_absolute_integer_total': maximum,
        'scope': 'CPU official THRML attention-component check at8320/2080/16448/4112positions, T8192 and128normchannels, exact independentint64 reconstruction and signed high-magnitude mirrors. Not GPU validation, full-model forward fidelity, quality, heldout scoring, physical bit flips, energy or latency.'}
    with (out/'independent_audit.json').open('x') as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
