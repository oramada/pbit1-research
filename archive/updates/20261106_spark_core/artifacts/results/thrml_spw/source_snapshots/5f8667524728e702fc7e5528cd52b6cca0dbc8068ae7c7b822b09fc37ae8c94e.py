"""CPU direct-command export validation, including actual trained controls."""
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
import torch
from .probability_tensor import ProbabilityTensor
from .program_export import export_program, load_program, operator_layout
from .probability_training import ProbabilityLinear


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    torch.set_num_threads(2)
    root = Path('results/thrml_spw/lfm3_1p7b')
    out = root / 'program_export_preflight'
    out.mkdir(exist_ok=True)
    names = ['program_export.py', 'programmed_worker.py', 'probability_tensor.py', 'preflight_program_export.py', 'probability_training.py']
    protocol = {'sources_sha256': {n: sha(Path(__file__).with_name(n)) for n in names},
                'samples': [1, 16], 'field_bits': 12, 'activation_bits': 12, 'key': 202610315,
                'scope': 'Canonical UINT16 commands with12informationbits plus fixedFP32sharedranges. Direct officialCPU THRML execution without reconstructing/re-encoding weights, exact public replay. Four operator kinds, initial/perturbedcontrols and actual trained projectionpilot. Not full-model trained export, language quality or physical memory/latency/energy claim.'}
    with (out / 'protocol.json').open('x') as f:
        json.dump(protocol, f, indent=2)
    programs, cases = [], []
    for kind, index in [('linear', 0), ('conv', 4), ('embedding', 8), ('elementwise', 12)]:
        source = root / 'group128_preflight' / f'input_{index}.npz'
        a = np.load(source)
        w = a['w'].T if kind == 'linear' else a['w'].T[:, None, :] if kind == 'conv' else a['w'].reshape(2, 2048) if kind == 'embedding' else a['w'].reshape(-1)
        encoder = ProbabilityTensor(torch.from_numpy(w.copy()).to(torch.bfloat16))
        lower, span = encoder.lower.clone(), encoder.span.clone()
        for variant in ['initial', 'perturbed']:
            if variant == 'perturbed':
                with torch.no_grad():
                    encoder.theta.add_(torch.linspace(-.25, .25, encoder.theta.numel()).reshape(encoder.theta.shape))
                encoder.project()
            path = out / f'{kind}_{variant}_program.npz'
            export_program(encoder, path)
            program = load_program(path)
            np.testing.assert_array_equal(program['codes'], encoder.commands()[0].numpy())
            np.testing.assert_array_equal(program['lower'], lower.numpy())
            np.testing.assert_array_equal(program['span'], span.numpy())
            layout = operator_layout(program, kind)
            with torch.no_grad():
                mean, _ = encoder.moments()
            wm = mean.T if kind == 'linear' else mean[:, 0, :].T if kind == 'conv' else mean.reshape(1, -1)
            _, _, x = ProbabilityLinear.quantize_inputs(torch.from_numpy(a['x']))
            expected = (x @ wm if kind == 'linear' else (x * wm[None]).sum(1)).numpy()
            ip = out / f'{kind}_{variant}_input.npz'
            np.savez(ip, x=a['x'], **layout)
            programs.append({'kind': kind, 'variant': variant, 'program': str(path),
                             'input': str(ip), 'source_input': str(source),
                             'source_input_sha256': sha(source)})
            cases.append((kind, variant, ip, expected, [1, 16]))
            if kind == 'embedding':
                # Embedding lookup and head are views of the same program.
                selected = operator_layout(program, 'embedding', rows=np.asarray([1, 0]))
                head = operator_layout(program, 'linear')
                np.testing.assert_array_equal(selected['codes'].reshape(2, 2048), head['codes'].T[[1, 0]])
    trained_path = root / 'probability_training_projected/training_arrays.npz'
    a = np.load(trained_path)
    encoder = ProbabilityTensor(torch.from_numpy(a['weights'].T.copy()).to(torch.bfloat16))
    with torch.no_grad():
        encoder.theta.copy_(torch.from_numpy(a['after_fields'].reshape(128, 8).T.copy()))
    path = out / 'linear_trained_program.npz'
    export_program(encoder, path)
    program = load_program(path)
    np.testing.assert_array_equal(program['codes'].T, a['after_codes'].reshape(128, 8))
    layout = operator_layout(program, 'linear')
    ip = out / 'linear_trained_input.npz'
    np.savez(ip, x=a['inputs'], **layout)
    programs.append({'kind': 'linear', 'variant': 'trained', 'program': str(path),
                     'input': str(ip), 'source_input': str(trained_path), 'source_input_sha256': sha(trained_path)})
    cases.append(('linear', 'trained', ip, a['after_mean'], [16]))
    log = (out / 'worker.log').open('w')
    worker = subprocess.Popen(['.venv-thrml/bin/python', '-m', 'research.thrml_spw.lfm3.programmed_worker'],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True, bufsize=1)
    assert json.loads(worker.stdout.readline())['ready']
    rows = []
    try:
        for kind, variant, ip, expected, sample_counts in cases:
            for samples in sample_counts:
                for mode in ['mean', 'native']:
                    call = len(rows)
                    op = out / f'output_{call}.npy'
                    request = {'input': str(ip), 'output': str(op), 'kind': kind, 'samples': samples,
                        'field_bits': 12, 'activation_bits': 12, 'key': protocol['key'], 'call': call,
                        'mode': mode, 'replay': mode == 'native', 'state_dir': str(out / 'raw_states')}
                    worker.stdin.write(json.dumps(request) + '\n'); worker.stdin.flush()
                    line = worker.stdout.readline()
                    if not line:
                        raise RuntimeError('Worker stopped; preserve logs')
                    receipt = json.loads(line)
                    assert receipt['ok'] and receipt['finite'] and receipt['program_codes_used_directly'] and not receipt['reference_weights_supplied']
                    if mode == 'mean':
                        np.testing.assert_allclose(np.load(op), expected, rtol=2e-12, atol=2e-12)
                    rows.append({'kind': kind, 'variant': variant, 'mode': mode, 'samples': samples,
                                 'input': str(ip), 'output': str(op), **receipt})
    finally:
        if worker.poll() is None:
            worker.stdin.write('{"stop":true}\n'); worker.stdin.flush(); worker.stdin.close()
        worker.wait(timeout=60); log.close()
    artifacts = [*out.glob('*_program.npz'), *out.glob('*_input.npz'), *out.glob('output_*.npy'), *(out / 'raw_states').glob('*.npz')]
    with (out / 'results.json').open('x') as f:
        json.dump({'status': 'all34_direct_program_cases_passed', 'programs': programs, 'receipts': rows,
                   'artifacts_sha256': {str(p): sha(p) for p in sorted(artifacts)}}, f, indent=2)
    print('All34 direct command cases passed', flush=True)


if __name__ == '__main__':
    main()
