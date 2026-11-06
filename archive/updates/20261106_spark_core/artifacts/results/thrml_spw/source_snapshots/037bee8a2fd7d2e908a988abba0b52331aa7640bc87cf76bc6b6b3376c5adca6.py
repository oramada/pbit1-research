"""Check unchanged gradients, separate tied uses, variance algebra and budget."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from .preflight_same_fit_control import Fixture
from .probability_operators import install
from .layer_allocation import instrument, variance_coefficient, allocate


def main():
    torch.set_num_threads(2)
    torch.manual_seed(202610390)
    model = Fixture().eval()
    state, encoders, aliases = install(model)
    ids = torch.tensor([[0, 2, 31, 2, 15], [15, 3, 2, 0, 1]])
    y = model(ids)
    y.float().square().mean().backward()
    base_logits = y.detach().clone()
    base_gradients = {name: encoder.theta.grad.clone() for name, encoder in encoders.items()}
    records = []
    def capture(owner, canonical, encoder, rows, gradient):
        value, norm = variance_coefficient(encoder, rows, gradient)
        codes = encoder.commands(rows)[0].numpy()
        field = codes.astype(np.float32) * np.float32(8 / 4095) - np.float32(4)
        p = np.exp(-np.logaddexp(0, -2 * field.astype(np.float64)))
        span = encoder.span.numpy() if rows is None else encoder.span.numpy()[rows.numpy()]
        qshape = (span.shape[0], span.shape[1], -1)
        g = gradient.detach().numpy().reshape(qshape)
        expected = float(np.sum(g * g * span.astype(np.float64) ** 2 * p.reshape(qshape) * (1 - p.reshape(qshape))))
        np.testing.assert_allclose(value, expected, rtol=2e-12, atol=2e-12)
        np.testing.assert_allclose(norm, np.square(g).sum(), rtol=2e-12, atol=2e-12)
        records.append({'module': owner, 'canonical': canonical, 'variance': value,
                        'gathered_lookup': rows is not None})
        return gradient
    originals = instrument(encoders, aliases, {'embedding'}, capture)
    model.zero_grad(set_to_none=True)
    y = model(ids)
    assert torch.equal(y, base_logits)
    y.float().square().mean().backward()
    assert len(records) == len(aliases) == 5
    assert {v['module'] for v in records} == set(aliases)
    assert next(v for v in records if v['module'] == 'embedding')['gathered_lookup']
    assert not next(v for v in records if v['module'] == 'head')['gathered_lookup']
    for name, encoder in encoders.items():
        assert torch.equal(encoder.theta.grad, base_gradients[name])
        encoder.weights = originals[name]
    scores = {v['module']: v['variance'] for v in records}
    costs = {'embedding': 1024, 'head': 4096, 'norm': 128, 'conv': 896, 'projection': 16384}
    proposal = allocate(scores, costs, 'embedding')
    assert proposal['selected_cost'] <= proposal['cost_cap'] and proposal['greedy_cost'] <= proposal['cost_cap']
    assert proposal['selected_samples']['embedding'] == 1024
    assert all(v in [16, 32, 64, 128, 256, 512, 1024, 2048, 4096] for v in proposal['selected_samples'].values())
    assert proposal['selected_proxy'] <= proposal['uniform_proxy']
    assert allocate({k: 0. for k in scores}, costs, 'embedding')['selected_proxy'] == 0
    report = {'status': 'instrumented_logits_gradients_tied_use_variances_and_budget_invariants_verified',
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'helper_sha256': hashlib.sha256(Path(__file__).with_name('layer_allocation.py').read_bytes()).hexdigest(),
        'module_uses': records, 'proposal': proposal,
        'scope': 'CPU fixture; unchanged outputs/allcontrolgradients bitwise, separate embedding/head gradient-use callbacks, independent NumPy scalarvariance algebra and allocation constraints. No full-model sensitivity/quality or native allocation claim.'}
    path = Path('results/thrml_spw/lfm3_1p7b/layer_allocation_preflight.json')
    with path.open('x') as f:
        json.dump(report, f, indent=2)
    print(json.dumps({'status': report['status'], 'uses': len(records), 'selected_policy': proposal['selected_policy']}))


if __name__ == '__main__':
    main()
