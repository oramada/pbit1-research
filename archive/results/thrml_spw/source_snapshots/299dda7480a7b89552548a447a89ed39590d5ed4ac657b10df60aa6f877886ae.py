"""GPU public-state replay at the largest proposed signed ACT12 sample count."""
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path('results/thrml_spw/lfm3_1p7b')
    out = root / 'sample8192_preflight'
    out.mkdir(exist_ok=True)
    protocol = {'sources_sha256': {n: sha(Path(__file__).with_name(n)) for n in
        ['preflight_sample8192.py', 'verify_sample8192.py', 'programmed_worker.py']},
        'samples': 8192, 'fanin': 128, 'real_outputs': 8, 'padded_outputs': 64, 'key': 202610391,
        'conditions': ['positive', 'negative'], 'fields': '+4forrealcolumns, dummyzero-spanpadding',
        'counter_bound': 2047 * 128 * 8192,
        'scope': 'Synthetic worst-magnitude ACT12/nativeint32edge test, directfinitecommands, officialGPU THRML andexactpublictrajectoryreplay. Samekeys forsignmirrors. Not fullneuralquality, performance orhardwarevalidation.'}
    assert protocol['counter_bound'] < np.iinfo(np.int32).max
    with (out / 'protocol.json').open('x') as f:
        json.dump(protocol, f, indent=2)
    log = (out / 'worker.log').open('x')
    worker = subprocess.Popen(['.venv/bin/python', '-m', 'research.thrml_spw.gpu_environment', '--',
        '.venv-thrml-gpu/bin/python', '-m', 'research.thrml_spw.lfm3.programmed_worker'],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True, bufsize=1)
    ready = json.loads(worker.stdout.readline())
    assert ready['ready'] and any('cuda' in v.lower() or 'gpu' in v.lower() for v in ready['devices'])
    records = []
    try:
        for condition, sign in [('positive', 1), ('negative', -1)]:
            directory = out / condition
            directory.mkdir()
            ip, op = directory / 'input.npz', directory / 'output.npy'
            np.savez(ip, x=np.full((1, 128), sign, np.float32), codes=np.full((128, 8), 4095, np.uint16),
                     lower=np.zeros((1, 8), np.float32), span=np.ones((1, 8), np.float32))
            request = {'input': str(ip), 'output': str(op), 'kind': 'linear', 'samples': 8192,
                'field_bits': 12, 'activation_bits': 12, 'key': protocol['key'], 'call': 0,
                'mode': 'native', 'replay': True, 'state_dir': str(directory / 'raw')}
            worker.stdin.write(json.dumps(request) + '\n'); worker.stdin.flush()
            line = worker.stdout.readline()
            if not line:
                raise RuntimeError('8192worker stopped; preserve logs')
            receipt = json.loads(line)
            assert receipt['ok'] and receipt['finite'] and receipt['public_replay_checked']
            records.append({'condition': condition, 'input': str(ip), 'input_sha256': sha(ip),
                'output': str(op), 'output_sha256': sha(op), 'raw': str(directory / 'raw/call0_tile0.npz'), **receipt})
        np.testing.assert_array_equal(np.load(records[0]['output']), -np.load(records[1]['output']))
    finally:
        if worker.poll() is None:
            worker.stdin.write('{"stop":true}\n'); worker.stdin.flush(); worker.stdin.close()
        worker.wait(timeout=60); log.close()
    for rec in records:
        rec['raw_sha256'] = sha(Path(rec['raw']))
    with (out / 'results.json').open('x') as f:
        json.dump({'status': 'both_T8192_signed_counter_edge_cases_publicly_replayed', 'worker_ready': ready, 'cases': records}, f, indent=2)
    print('Both signed T8192 edge cases passed native/public replay')


if __name__ == '__main__':
    main()
