"""Forward/gradient/saved-state parity on real tensor kinds and edge cases."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from .probability_tensor import ProbabilityTensor
from .recomputed_probability import RecomputedProbabilityTensor


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def saved_state(module, weight):
    persistent = {v.untyped_storage().data_ptr() for v in [module.theta, module.lower, module.span]}
    saved = {}
    def pack(tensor):
        storage = tensor.untyped_storage()
        saved[(storage.data_ptr(), storage.nbytes())] = {'dtype': str(tensor.dtype), 'bytes': storage.nbytes(),
            'persistent': storage.data_ptr() in persistent}
        return tensor
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda value: value):
        output = module(weight)
    gradient = torch.autograd.grad(output, module.theta, grad_outputs=torch.ones_like(output))[0]
    return {'unique_saved_storage_bytes': sum(v['bytes'] for v in saved.values()),
            'transient_saved_storage_bytes': sum(v['bytes'] for v in saved.values() if not v['persistent']),
            'saved': list(saved.values())}, gradient


def main():
    torch.set_num_threads(2)
    root = Path('results/thrml_spw/lfm3_1p7b')
    out = root / 'recomputed_probability_preflight'
    out.mkdir(exist_ok=True)
    protocol = {'sources_sha256': {n: sha(Path(__file__).with_name(n)) for n in
        ['probability_tensor.py', 'recomputed_probability.py', 'preflight_recomputed_probability.py']},
        'variants': ['initial', 'perturbed', 'endpoint_and_outside'],
        'scope': 'CPU first-order declared projected STE parity, real operator tensor shapes, zero-span case, gathered rows and saved-storage inspection. No native-sampler gradient, full-model backward, training/quality, total peak-memory or hardware benefit claim.'}
    with (out / 'protocol.json').open('x') as f:
        json.dump(protocol, f, indent=2)
    cases = []
    for kind, index in [('linear', 0), ('conv', 4), ('embedding', 8), ('elementwise', 12)]:
        a = np.load(root / 'group128_preflight' / f'input_{index}.npz')
        w = a['w'].T if kind == 'linear' else a['w'].T[:, None, :] if kind == 'conv' else a['w'].reshape(2, 2048) if kind == 'embedding' else a['w'].reshape(-1)
        cases.append((kind, torch.from_numpy(w.copy()).to(torch.bfloat16)))
    cases.append(('zero_span', torch.ones((3, 1, 7), dtype=torch.bfloat16)))
    rows, arrays = [], {}
    for kind, weight in cases:
        for variant in protocol['variants']:
            reference, recomputed = ProbabilityTensor(weight), RecomputedProbabilityTensor(weight)
            with torch.no_grad():
                if variant == 'perturbed':
                    reference.theta.add_(torch.linspace(-.25, .25, weight.numel()).reshape(weight.shape))
                    reference.project()
                elif variant == 'endpoint_and_outside':
                    values = torch.tensor([-5., -4., 0., 4., 5.])
                    reference.theta.copy_(values[torch.arange(weight.numel()) % 5].reshape(weight.shape))
                recomputed.theta.copy_(reference.theta)
            ref_mean, ref_var = reference.moments()
            new_mean, new_var = recomputed.moments()
            torch.testing.assert_close(new_mean, ref_mean, rtol=0, atol=0)
            torch.testing.assert_close(new_var, ref_var, rtol=0, atol=0)
            target = torch.linspace(-.5, .5, weight.numel()).reshape(weight.shape).double()
            ref_gradient = torch.autograd.grad(((ref_mean - target).square() + .7 * ref_var).mean(), reference.theta)[0]
            new_gradient = torch.autograd.grad(((new_mean - target).square() + .7 * new_var).mean(), recomputed.theta)[0]
            torch.testing.assert_close(new_gradient, ref_gradient, rtol=2e-6, atol=1e-10)
            ref_var_only = reference.moments()[1].mean()
            new_var_only = recomputed.variance_mean()
            torch.testing.assert_close(new_var_only, ref_var_only, rtol=0, atol=0)
            g0 = torch.autograd.grad(ref_var_only, reference.theta)[0]
            g1 = torch.autograd.grad(new_var_only, recomputed.theta)[0]
            torch.testing.assert_close(g1, g0, rtol=2e-6, atol=1e-10)
            torch.testing.assert_close(recomputed(weight), reference(weight), rtol=0, atol=0)
            before, gb = saved_state(reference, weight)
            after, ga = saved_state(recomputed, weight)
            torch.testing.assert_close(ga, gb, rtol=2e-6, atol=1e-10)
            assert after['transient_saved_storage_bytes'] == 0
            assert before['transient_saved_storage_bytes'] >= weight.numel() * 8
            label = kind + '_' + variant
            arrays.update({label + '_theta': reference.theta.detach().numpy(), label + '_lower': reference.lower.numpy(),
                           label + '_span': reference.span.numpy(), label + '_mean': new_mean.detach().numpy(),
                           label + '_variance': new_var.detach().numpy(), label + '_gradient': new_gradient.numpy(),
                           label + '_variance_gradient': g1.numpy()})
            rows.append({'kind': kind, 'variant': variant, 'shape': list(weight.shape), 'reference_saved_state': before,
                         'recomputed_saved_state': after})
            if kind == 'embedding':
                indices = torch.tensor([1, 0, 1])
                e0 = reference.moments(rows=indices)[0]
                e1 = recomputed.moments(rows=indices)[0]
                torch.testing.assert_close(e1, e0, rtol=0, atol=0)
                g0 = torch.autograd.grad(e0.sum(), reference.theta)[0]
                g1 = torch.autograd.grad(e1.sum(), recomputed.theta)[0]
                torch.testing.assert_close(g1, g0, rtol=2e-6, atol=1e-10)
    np.savez_compressed(out / 'arrays.npz', **arrays)
    with (out / 'results.json').open('x') as f:
        json.dump({'status': 'all15_cases_forward_first_order_gradient_and_saved_state_checks_passed',
                   'cases': rows, 'arrays_sha256': sha(out / 'arrays.npz')}, f, indent=2)
    print('All15 recomputed backward cases passed')


if __name__ == '__main__':
    main()
