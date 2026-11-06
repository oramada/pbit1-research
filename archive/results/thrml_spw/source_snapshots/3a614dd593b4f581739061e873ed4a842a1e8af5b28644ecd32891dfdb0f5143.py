"""CPU request/response and raw-evidence fixture for the tiled word worker."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from .adapted_word_conversion import convert_codes


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    names = ['adapted_word_worker.py', 'preflight_adapted_word_worker.py', 'adapted_word_conversion.py', 'counted_bank.py']
    sources = {name: sha(Path(__file__).with_name(name)) for name in names}
    for digest in sources.values():
        assert sha(Path('results/thrml_spw/source_snapshots') / (digest + '.py')) == digest
    protocol = {'sources_sha256': sources,
        'vendor_thrml_commit': subprocess.check_output(['git', '-C', 'vendor/thrml', 'rev-parse', 'HEAD'], text=True).strip(),
        'kind': 'CPU fixture only; no model scoring or bias fitting',
        'cases': [('linear', 16, 65, 4, 16, 3), ('linear', 128, 257, 8, 16, 2),
                  ('conv', 7, 67, 4, 16, 3), ('elementwise', 1, 257, 8, 16, 3),
                  ('embedding', 1, 65, 4, 16, 3), ('linear', 7680, 5, 8, 2, 2),
                  ('linear', 128, 64, 4, 1024, 2), ('conv', 7, 64, 8, 512, 2)]}
    assert protocol['vendor_thrml_commit'] == '2d54cd8404df85bb65703d7e8b4e84e48c1b72a9'
    with (args.output / 'protocol.json').open('x') as handle:
        json.dump(protocol, handle, indent=2)
    log = (args.output / 'worker.log').open('x')
    worker = subprocess.Popen([sys.executable, '-m', 'research.thrml_spw.lfm3.adapted_word_worker'],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True, bufsize=1)
    records = []
    try:
        ready = json.loads(worker.stdout.readline())
        assert ready['ready'] and all('cpu' in name.lower() for name in ready['devices'])
        rng = np.random.default_rng(202610440)
        for ordinal, (kind, fanin, channels, bits, samples, batch) in enumerate(protocol['cases']):
            folder = args.output / f'case{ordinal}'
            folder.mkdir()
            depthwise = kind != 'linear'
            shape = (batch, fanin, channels) if depthwise else (batch, fanin)
            x = rng.normal(0, .2, size=shape).astype(np.float32)
            x[0] = 1
            x[1] = -1
            if kind == 'embedding':
                x[:] = 1
            if kind == 'elementwise':
                x[:] = 0
            words = convert_codes(rng.integers(0, 4096, size=(fanin, channels), dtype=np.uint16), bits)
            groups = fanin // min(128, fanin)
            lower = rng.normal(0, .2, size=(groups, channels)).astype(np.float32)
            span = np.abs(rng.normal(.3, .1, size=(groups, channels))).astype(np.float32)
            span[:, 0] = 0
            input_path = folder / 'input.npz'
            with input_path.open('xb') as handle:
                np.savez_compressed(handle, x=x, words=words, lower=lower, span=span)
            for mode in ['mean', 'native']:
                call = 2 * ordinal + int(mode == 'native')
                output = folder / f'{mode}.npy'
                request = {'input': str(input_path), 'output': str(output), 'kind': kind,
                           'word_bits': bits, 'samples': samples, 'activation_bits': 12,
                           'key': 202610440 + ordinal, 'call': call, 'mode': mode,
                           'replay': mode == 'native', 'state_dir': str(folder / 'states')}
                worker.stdin.write(json.dumps(request) + '\n')
                worker.stdin.flush()
                line = worker.stdout.readline()
                if not line:
                    raise RuntimeError('Worker terminated; preserve worker.log and partial evidence')
                receipt = json.loads(line)
                assert receipt['ok'] and receipt['finite']
                traces = [{'path': str(p), 'sha256': sha(p)} for p in sorted((folder / 'states').glob('*.npz'))] if mode == 'native' else []
                records.append({'request': request, 'receipt': receipt, 'input_sha256': sha(input_path),
                                'output_sha256': sha(output), 'raw_traces': traces})
                (args.output / 'progress.json').write_text(json.dumps({'completed_requests': len(records), 'last_kind': kind, 'last_mode': mode}) + '\n')
                print(json.dumps({'completed_requests': len(records), 'kind': kind, 'fanin': fanin,
                                  'channels': channels, 'bits': bits, 'mode': mode}), flush=True)
    finally:
        if worker.poll() is None:
            worker.stdin.write('{"stop":true}\n')
            worker.stdin.flush()
            worker.stdin.close()
        code = worker.wait(timeout=60)
        log.close()
    assert code == 0 and len(records) == 16
    with (args.output / 'results.json').open('x') as handle:
        json.dump({'status': 'all_CPU_fixture_requests_complete_pending_independent_audit',
                   'protocol_sha256': sha(args.output / 'protocol.json'), 'worker_ready': ready, 'records': records}, handle, indent=2)
        handle.write('\n')


if __name__ == '__main__':
    main()
