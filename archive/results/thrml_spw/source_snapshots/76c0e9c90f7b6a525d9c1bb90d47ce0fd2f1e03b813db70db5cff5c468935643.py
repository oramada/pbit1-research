"""CPU validation of internal four-node operators on preserved LFM3 inputs."""
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path('results/thrml_spw/lfm3_1p7b')
    out = root / 'word_preflight'
    out.mkdir(exist_ok=True)
    inputs = {k: root / 'group128_preflight' / f'input_{i}.npz'
              for k, i in [('linear', 0), ('conv', 4), ('embedding', 8), ('elementwise', 12)]}
    protocol = {'source_sha256': sha(Path(__file__)),
                'worker_sha256': sha(Path(__file__).with_name('word_worker.py')),
                'observer_sha256': sha(Path(__file__).with_name('precision_worker.py')),
                'inputs_sha256': {str(p): sha(p) for p in inputs.values()},
                'samples': [1, 16], 'modes': ['mean', 'native'],
                'key': 202610311, 'pbits_per_parameter': 4,
                'scope': 'Internal positional four-free-node representation, finite endpoint fields +/-4. Actual saved LFM3 operator inputs. CPU official THRML, exact public trajectory replay. No full-model task quality or Extropic hardware-format claim.'}
    with (out / 'protocol.json').open('x') as f:
        json.dump(protocol, f, indent=2)
    log = (out / 'worker.log').open('w')
    worker = subprocess.Popen(['.venv-thrml/bin/python', '-m', 'research.thrml_spw.lfm3.word_worker'],
                              stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True, bufsize=1)
    assert json.loads(worker.stdout.readline())['ready']
    rows = []
    try:
        for kind, path in inputs.items():
            a = np.load(path)
            w, x = a['w'].astype(np.float64), a['x']
            size = min(128, w.shape[0])
            grouped = w.reshape(-1, size, w.shape[1])
            lower = grouped.min(1)[:, None, :]
            span = np.ptp(grouped, axis=1)[:, None, :]
            if kind in ['embedding', 'elementwise']:
                lower, span = a['lower'][None, None, :], a['span'][None, None, :]
            p = np.divide(grouped - lower, span, out=np.full_like(grouped, .5), where=span != 0).clip(0, 1)
            word = np.rint(15 * p).astype(np.uint8)
            bits = ((word[..., None] >> np.arange(4)) & 1)
            q = np.where(bits, 1 / (1 + np.exp(-8.0)), 1 / (1 + np.exp(8.0)))
            mean_weight = (lower + span * np.sum(q * (2.0 ** np.arange(4) / 15), axis=-1)).reshape(w.shape)
            scale = float(np.abs(x).max()) / 2047 if np.any(x) else 1.0
            quantized = np.rint(x / scale).clip(-2047, 2047).astype(np.int32).astype(np.float64) * scale
            expected = quantized @ mean_weight if kind == 'linear' else np.sum(quantized * mean_weight[None, :, :], axis=1)
            for samples in protocol['samples']:
                for mode in protocol['modes']:
                    call = len(rows)
                    op = out / f'output_{call}.npy'
                    req = {'input': str(path), 'output': str(op), 'kind': kind,
                           'samples': samples, 'activation_bits': 12, 'key': protocol['key'],
                           'call': call, 'mode': mode, 'replay': mode == 'native',
                           'state_dir': str(out / 'raw_states')}
                    worker.stdin.write(json.dumps(req) + '\n')
                    worker.stdin.flush()
                    line = worker.stdout.readline()
                    if not line:
                        raise RuntimeError('Word worker stopped; preserve logs')
                    receipt = json.loads(line)
                    assert receipt['ok'] and receipt['finite'] and receipt['pbits_per_parameter'] == 4
                    if mode == 'mean':
                        np.testing.assert_allclose(np.load(op), expected, rtol=2e-12, atol=2e-12)
                    else:
                        assert receipt['public_replay_checked']
                    rows.append({'kind': kind, 'mode': mode, 'samples': samples,
                                 'input': str(path), 'output': str(op), **receipt})
            print(f'{kind} passed', flush=True)
    finally:
        if worker.poll() is None:
            worker.stdin.write('{"stop":true}\n')
            worker.stdin.flush()
            worker.stdin.close()
        worker.wait(timeout=60)
        log.close()
    artifacts = [*sorted((out / 'raw_states').glob('*.npz')), *sorted(out.glob('output_*.npy'))]
    with (out / 'results.json').open('x') as f:
        json.dump({'status': 'all16_operator_cases_passed', 'receipts': rows,
                   'artifacts_sha256': {str(p): sha(p) for p in artifacts}}, f, indent=2)


if __name__ == '__main__':
    main()
