"""Projected-gradient boundary treatment; separate controlled operator pilot."""
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
import torch
from .probability_training_projected import ProjectedProbabilityLinear as ProbabilityLinear


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    torch.set_num_threads(2)
    root = Path('results/thrml_spw/lfm3_1p7b')
    out = root / 'probability_training_projected'
    out.mkdir(exist_ok=True)
    source = root / 'group128_preflight/input_0.npz'
    names = ['probability_training_projected.py','probability_training.py', 'probability_training_worker.py', 'preflight_probability_training_projected.py', 'precision_worker.py']
    protocol = {'sources_sha256': {n: sha(Path(__file__).with_name(n)) for n in names},
                'input_sha256': sha(source), 'input_slice': 'First128fan-in/first8outputs; all16captured input positions.',
                'samples': 16, 'optimizer': 'Adam', 'learning_rate': .03, 'steps': 200,
                'field_bits': 12, 'field_bound': 4, 'activation_bits': 12, 'fixed_ranges': True, 'boundary_gradient':'Identity STE through clipping and quantization, with explicit projection after every step. Retrospective treatment motivated by first pilot boundary freezing; all other inputs/optimizer/native budgets unchanged.',
                'native_seeds': list(range(5)), 'trials_per_seed': 100, 'native_key': 202610314,
                'objective': 'Exact marginal E[mean(output-target)^2] for independent stochastic weights: squared mean error plus output variance/T. Fixed original quantized-input linear target.',
                'scope': 'CPU controlled1024-weight real-LFM3 projection slice. Biased control-rounding STE with expected-operator gradients, then separate official THRML before/after execution and full public replay. Shared trial keys across conditions for pairing. No full-model training, language quality, BF16 equivalence or hardware claim.'}
    with (out / 'protocol.json').open('x') as f:
        json.dump(protocol, f, indent=2)
    # A separate objective explicitly requires inward movement at both limits.
    toy=ProbabilityLinear(torch.linspace(-1,1,128,dtype=torch.float64)[:,None].expand(128,8).clone())
    toy_arrays={'before_theta':toy.theta.detach().numpy().copy(),'before_codes':toy.commands()[0].numpy()}
    toy_optimizer=torch.optim.SGD(toy.parameters(),lr=.1)
    for step in range(8):
        toy_optimizer.zero_grad();mu,_=toy.moments();toy_loss=mu.square().sum();toy_loss.backward()
        if step==0:toy_arrays['initial_gradient']=toy.theta.grad.detach().numpy().copy()
        toy_optimizer.step();toy.project()
    toy_arrays.update(after_theta=toy.theta.detach().numpy().copy(),after_codes=toy.commands()[0].numpy())
    np.savez_compressed(out/'boundary_arrays.npz',**toy_arrays)
    assert np.all(toy_arrays['after_codes'][0,0,:]>0) and np.all(toy_arrays['after_codes'][0,-1,:]<4095)
    a = np.load(source)
    w = torch.from_numpy(a['w'][:128, :8].astype(np.float64))
    x = torch.from_numpy(a['x'][:, :128])
    model = ProbabilityLinear(w)
    integers, scale, qx = model.quantize_inputs(x)
    target = qx @ w
    arrays = {'weights': w.numpy(), 'inputs': x.numpy(), 'input_integers': integers.numpy(),
              'input_scale': np.asarray(scale), 'target': target.numpy(),
              'lower': model.lower.numpy(), 'span': model.span.numpy()}
    optimizer = torch.optim.Adam(model.parameters(), lr=.03)
    history = []
    for step in range(201):
        optimizer.zero_grad()
        mean, variance = model(x, samples=16)
        loss = ((mean - target).square() + variance).mean()
        if step in [0, 200]:
            label = 'before' if step == 0 else 'after'
            codes, fields = model.commands()
            arrays.update({label + '_theta': model.theta.detach().numpy().copy(),
                           label + '_codes': codes.numpy(), label + '_fields': fields.numpy(),
                           label + '_mean': mean.detach().numpy(), label + '_variance': variance.detach().numpy()})
        loss.backward()
        if step == 0:
            arrays['initial_gradient'] = model.theta.grad.detach().numpy().copy()
        history.append({'step': step, 'expected_MSE': loss.item(),
                        'squared_mean_error': (mean - target).square().mean().item(),
                        'mean_variance': variance.mean().item()})
        if step < 200:
            optimizer.step()
            model.project()
    np.savez_compressed(out / 'training_arrays.npz', **arrays)
    with (out / 'training_results.json').open('x') as f:
        json.dump({'history': history, 'arrays_sha256': sha(out / 'training_arrays.npz')}, f, indent=2)
    # Compare the initial training surrogate with the already validated
    # finite-control mean worker, including its ACT12 input quantization.
    np.savez(out / 'mean_input.npz', x=x.numpy(), w=w.numpy())
    with (out / 'mean_worker.log').open('w') as log:
        worker = subprocess.Popen(['.venv-thrml/bin/python', '-m', 'research.thrml_spw.lfm3.precision_worker'],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True, bufsize=1)
        assert json.loads(worker.stdout.readline())['ready']
        request = {'input': str(out / 'mean_input.npz'), 'output': str(out / 'mean_reference.npy'),
                   'kind': 'linear', 'samples': 16, 'field_bits': 12, 'activation_bits': 12,
                   'key': 202610314, 'call': 0, 'mode': 'mean'}
        worker.stdin.write(json.dumps(request) + '\n'); worker.stdin.flush()
        receipt = json.loads(worker.stdout.readline())
        assert receipt['ok'] and receipt['finite']
        worker.stdin.write('{"stop":true}\n'); worker.stdin.flush(); worker.stdin.close()
        assert worker.wait(timeout=60) == 0
    np.testing.assert_allclose(arrays['before_mean'], np.load(out / 'mean_reference.npy'), rtol=2e-12, atol=2e-12)
    (out / 'mean_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    with (out / 'worker.log').open('w') as log:
        subprocess.run(['.venv-thrml/bin/python', '-m', 'research.thrml_spw.lfm3.probability_training_worker', str(out)], stdout=log, stderr=subprocess.STDOUT, check=True)
    print(json.dumps({'before': history[0], 'after': history[-1], 'native_status': json.loads((out / 'native_results.json').read_text())['status']}))


if __name__ == '__main__':
    main()
