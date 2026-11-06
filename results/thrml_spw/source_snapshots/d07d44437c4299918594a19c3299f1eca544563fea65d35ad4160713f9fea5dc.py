"""Official THRML CPU worker at prospective 1,028-position holdout shapes."""
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    out = Path('results/thrml_spw/lfm3_1p7b/holdout_worker_cpu_preflight')
    out.mkdir(exist_ok=False)
    names = ['preflight_holdout_worker_cpu.py', 'verify_holdout_worker_cpu.py',
             'counted_programmed_worker.py', 'counted_bank.py']
    auxiliary = ['research/thrml_spw/native.py', 'research/thrml_spw/controlled_positive_native.py']
    protocol = {'sources_sha256': {n: sha(Path(__file__).with_name(n)) for n in names},
        'auxiliary_sources_sha256': {n: sha(Path(n)) for n in auxiliary},
        'vendor_thrml_git_commit': '2d54cd8404df85bb65703d7e8b4e84e48c1b72a9',
        'cases': [[128, 65, 8192, 'linear'], [7680, 65, 16, 'linear'],
                  [7, 65, 16, 'conv'], [1, 65, 8192, 'elementwise']],
        'positions': 1028, 'key': 202610703,
        'scope': 'Synthetic worker shape/public-state/integer fidelity only. CPU official THRML, no model weights or reserved text. Not full-model inference or GPU shape validation.'}
    with (out / 'protocol.json').open('x') as f:
        json.dump(protocol, f, indent=2)
    rng = np.random.default_rng(protocol['key'])
    records = []
    with (out / 'worker.log').open('x') as log:
        worker = subprocess.Popen(['.venv-thrml/bin/python', '-m', 'research.thrml_spw.lfm3.counted_programmed_worker'],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True, bufsize=1)
        try:
            ready = json.loads(worker.stdout.readline())
            assert ready['ready'] and all('cpu' in device.lower() for device in ready['devices'])
            for index, (fanin, channels, samples, kind) in enumerate(protocol['cases']):
                size = min(128, fanin)
                shape = (1028, fanin) if kind == 'linear' else (1028, fanin, channels)
                x = rng.integers(-2047, 2048, shape).astype(np.float32)
                x[0] = 2047; x[1] = -2047
                codes = rng.integers(0, 4096, (fanin, channels), dtype=np.uint16)
                if index == 0:
                    codes[:, :64] = 4095
                    codes[:, 64] = 0
                else:
                    codes[:, 0] = 0; codes[:, 1] = 4095
                lower = rng.integers(-8, 9, (fanin//size, channels)).astype(np.float32)/16
                span = rng.integers(1, 9, lower.shape).astype(np.float32)/16
                ip, op = out/f'input{index}.npz', out/f'output{index}.npy'
                with ip.open('xb') as f:
                    np.savez_compressed(f, x=x, codes=codes, lower=lower, span=span)
                req = {'input': str(ip), 'output': str(op), 'kind': kind, 'samples': samples,
                    'field_bits': 12, 'activation_bits': 12, 'key': protocol['key'],
                    'call': index, 'mode': 'native', 'replay': True, 'state_dir': str(out/'states')}
                worker.stdin.write(json.dumps(req)+'\n'); worker.stdin.flush()
                line = worker.stdout.readline()
                if not line:
                    raise RuntimeError('CPU worker ended; preserve log')
                receipt = json.loads(line)
                assert receipt['ok'] and receipt['finite'] and receipt['public_replay_checked']
                records.append({'request': req, 'receipt': receipt})
                print('completed case', index, kind, fanin, samples, flush=True)
        finally:
            if worker.poll() is None:
                worker.stdin.write('{"stop":true}\n'); worker.stdin.flush(); worker.stdin.close()
            assert worker.wait(timeout=60) == 0
    artifacts = [*out.glob('input*.npz'), *out.glob('output*.npy'), *out.glob('states/*.npz')]
    result = {'status': 'all_four_1028_position_CPU_worker_cases_produced', 'worker_ready': ready,
        'protocol_sha256': sha(out/'protocol.json'), 'records': records,
        'artifacts_sha256': {str(p): sha(p) for p in sorted(artifacts)}}
    with (out/'results.json').open('x') as f:
        json.dump(result, f, indent=2)


if __name__ == '__main__':
    main()
