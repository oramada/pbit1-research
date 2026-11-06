"""Compare new worker to all34 accepted direct-program cases plus wide tiles."""
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path('results/thrml_spw/lfm3_1p7b'); out = root / 'batched_worker_preflight'
    out.mkdir(exist_ok=True)
    old = root / 'program_export_gpu_preflight'
    old_result = json.loads((old / 'results.json').read_text())
    old_audit = json.loads((old / 'independent_audit.json').read_text())
    assert old_audit['status'] == 'all34_direct_command_serialization_mapping_public_counters_and_readouts_verified'
    protocol = {'sources_sha256': {n: sha(Path(__file__).with_name(n)) for n in
        ['batched_programmed_worker.py', 'batched_bank.py', 'programmed_worker.py', 'preflight_batched_worker.py', 'verify_batched_worker.py']},
        'original_GPU_results_sha256': sha(old / 'results.json'), 'original_GPU_audit_sha256': sha(old / 'independent_audit.json'),
        'bank_audit_sha256': sha(root / 'batched_bank_preflight/independent_audit.json'),
        'scope': 'Prospective component-only fidelity, not adoption in any existing fullmodel protocol or timing/quality claim.'}
    with (out / 'protocol.json').open('x') as f: json.dump(protocol, f, indent=2)
    logs, workers = [], {}
    for name, module in [('new', 'batched_programmed_worker'), ('reference', 'programmed_worker')]:
        log = (out / (name + '.log')).open('x'); logs.append(log)
        w = subprocess.Popen(['.venv/bin/python', '-m', 'research.thrml_spw.gpu_environment', '--',
            '.venv-thrml-gpu/bin/python', '-m', 'research.thrml_spw.lfm3.' + module],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True, bufsize=1)
        assert json.loads(w.stdout.readline())['ready']; workers[name] = w
    records = []
    def request(name, req):
        w = workers[name]; w.stdin.write(json.dumps(req) + '\n'); w.stdin.flush()
        line = w.stdout.readline()
        if not line: raise RuntimeError(f'{name} worker stopped; preserve logs')
        receipt = json.loads(line); assert receipt['ok'] and receipt['finite']
        return receipt
    try:
        for rec in old_result['receipts']:
            ip = Path(rec['input']); assert sha(ip) == old_result['artifacts_sha256'][str(ip)]
            op = out / f'output{rec["call"]}.npy'
            req = {'input': str(ip), 'output': str(op), 'kind': rec['kind'], 'samples': rec['samples'],
                'field_bits': 12, 'activation_bits': 12, 'key': 202610315, 'call': rec['call'], 'mode': rec['mode'],
                'replay': rec['mode'] == 'native', 'state_dir': str(out / 'new_states')}
            receipt = request('new', req)
            reference = Path(rec['output']); assert sha(reference) == old_result['artifacts_sha256'][str(reference)]
            np.testing.assert_array_equal(np.load(op), np.load(reference))
            records.append({'request': req, 'receipt': receipt, 'reference_output': str(reference), 'reference_states': str(old / 'raw_states')})
            print('prior case', rec['call'], 'exact', flush=True)
        rng = np.random.default_rng(202610411)
        for j, (fanin, channels, samples, kind) in enumerate([(128, 323, 16, 'linear'),
                (7680, 129, 16, 'linear'), (128, 197, 1024, 'linear'), (1, 323, 8192, 'elementwise')]):
            groups = fanin // min(fanin, 128)
            shape = (3, fanin) if kind == 'linear' else (3, 1, channels)
            x = rng.normal(size=shape).astype(np.float32)
            codes = rng.integers(0, 4096, (fanin, channels), dtype=np.uint16)
            codes.flat[0] = 0; codes.flat[-1] = 4095
            lower = rng.normal(size=(groups, channels)).astype(np.float32)
            span = rng.uniform(.001, 1., size=lower.shape).astype(np.float32)
            ip = out / f'wide{j}.npz'; np.savez(ip, x=x, codes=codes, lower=lower, span=span)
            req = {'input': str(ip), 'kind': kind, 'samples': samples, 'field_bits': 12, 'activation_bits': 12,
                'key': 202610411, 'call': 100 + j, 'mode': 'native', 'replay': True}
            receipts = {}
            for name in ['reference', 'new']:
                req.update(output=str(out / f'{name}_wide{j}.npy'), state_dir=str(out / (name + '_states')))
                receipts[name] = request(name, req)
            np.testing.assert_array_equal(np.load(out / f'new_wide{j}.npy'), np.load(out / f'reference_wide{j}.npy'))
            records.append({'request': dict(req), 'receipt': receipts['new'], 'reference_output': str(out / f'reference_wide{j}.npy'),
                'reference_states': str(out / 'reference_states'), 'reference_receipt': receipts['reference']})
            print('wide case', j, 'exact', flush=True)
    finally:
        for w in workers.values():
            if w.poll() is None:
                w.stdin.write('{"stop":true}\n'); w.stdin.flush(); w.stdin.close()
            assert w.wait(timeout=60) == 0
        for log in logs: log.close()
    artifacts = [*out.glob('*.npy'), *out.glob('wide*.npz'), *out.glob('*_states/*.npz')]
    result = {'status': 'all38_batched_worker_outputs_match_reference_producer', 'records': records,
        'protocol_sha256': sha(out / 'protocol.json'), 'artifacts_sha256': {str(p): sha(p) for p in sorted(artifacts)}}
    with (out / 'results.json').open('x') as f: json.dump(result, f, indent=2)


if __name__ == '__main__':
    main()
