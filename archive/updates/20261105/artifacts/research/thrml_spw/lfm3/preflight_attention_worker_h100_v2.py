"""GPU replay of frozen synthetic long-shape CPU inputs; no model or text."""
import hashlib
import json
from pathlib import Path
import subprocess


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path('results/thrml_spw/lfm3_1p7b')
    cpu = root / 'attention_worker_cpu_preflight_v2'
    out = root / 'attention_worker_h100_preflight_v2'
    out.mkdir(exist_ok=False)
    cp = json.loads((cpu/'protocol.json').read_text())
    cr = json.loads((cpu/'results.json').read_text())
    ca = json.loads((cpu/'independent_audit.json').read_text())
    assert ca['status'] == 'all_four_attention_expanded_public_states_int64_counters_F64_readouts_and_receipts_verified'
    assert ca['results_sha256'] == sha(cpu/'results.json') and cr['protocol_sha256'] == sha(cpu/'protocol.json')
    for name, digest in cr['artifacts_sha256'].items():
        assert sha(Path(name)) == digest
    names = ['preflight_attention_worker_h100_v2.py', 'verify_attention_worker_h100_v2.py', 'counted_programmed_worker.py', 'counted_bank.py']
    protocol = {'sources_sha256': {n: sha(Path(__file__).with_name(n)) for n in names},
        'auxiliary_sources_sha256': cp['auxiliary_sources_sha256'], 'cases': cp['cases'],
        'vendor_thrml_git_commit': cp['vendor_thrml_git_commit'], 'positions_by_case': cp['positions_by_case'], 'key': cp['key'],
        'CPU_reference_results_sha256': sha(cpu/'results.json'),
        'CPU_reference_audit_sha256': sha(cpu/'independent_audit.json'),
        'scope': 'Synthetic GPU long-shape public-state/int64 fidelity with a frozen same-key CPU reference. Cross-backend equality is measured, not assumed. No model weights or reserved text.'}
    for name, digest in protocol['auxiliary_sources_sha256'].items():
        assert sha(Path(name)) == digest
    with (out/'protocol.json').open('x') as f:
        json.dump(protocol, f, indent=2)
    records = []
    with (out/'worker.log').open('x') as log:
        worker = subprocess.Popen(['.venv/bin/python', '-m', 'research.thrml_spw.gpu_environment', '--',
            '.venv-thrml-gpu/bin/python', '-m', 'research.thrml_spw.lfm3.counted_programmed_worker'],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True, bufsize=1)
        try:
            ready = json.loads(worker.stdout.readline())
            assert ready['ready'] and any('gpu' in v.lower() or 'cuda' in v.lower() for v in ready['devices'])
            for index, reference in enumerate(cr['records']):
                req = {**reference['request'], 'output': str(out/f'output{index}.npy'), 'state_dir': str(out/'states')}
                worker.stdin.write(json.dumps(req)+'\n'); worker.stdin.flush()
                line = worker.stdout.readline()
                if not line:
                    raise RuntimeError('GPU worker ended; preserve logs')
                receipt = json.loads(line)
                assert receipt['ok'] and receipt['finite'] and receipt['public_replay_checked']
                records.append({'request': req, 'receipt': receipt})
                print('completed GPU case', index, flush=True)
        finally:
            if worker.poll() is None:
                worker.stdin.write('{"stop":true}\n'); worker.stdin.flush(); worker.stdin.close()
            assert worker.wait(timeout=60) == 0
    artifacts = [*[Path(v['request']['input']) for v in records], *out.glob('output*.npy'), *out.glob('states/*.npz')]
    result = {'status': 'all_four_attention_expanded_GPU_worker_cases_produced', 'worker_ready': ready,
        'protocol_sha256': sha(out/'protocol.json'), 'records': records,
        'artifacts_sha256': {str(p): sha(p) for p in sorted(artifacts)}}
    with (out/'results.json').open('x') as f:
        json.dump(result, f, indent=2)


if __name__ == '__main__':
    main()
