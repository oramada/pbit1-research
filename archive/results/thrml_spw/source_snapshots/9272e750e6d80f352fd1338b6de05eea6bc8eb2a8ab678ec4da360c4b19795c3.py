"""ACT12 differentiable operator checks against preserved public THRML draws."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from .probability_operators import OperatorProbabilityTensor, execute, quantize_act12


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    torch.set_num_threads(2)
    root = Path('results/thrml_spw/lfm3_1p7b')
    out = root / 'probability_operator_preflight'
    out.mkdir(exist_ok=True)
    prior = root / 'program_export_preflight'
    audit = json.loads((prior / 'independent_audit.json').read_text())
    assert audit['status'] == 'all34_direct_command_serialization_mapping_public_counters_and_readouts_verified'
    reference = json.loads((prior / 'results.json').read_text())
    protocol = {'sources_sha256': {n: sha(Path(__file__).with_name(n)) for n in
                ['probability_operators.py', 'probability_tensor.py', 'recomputed_probability.py', 'preflight_probability_operators.py']},
                'reference_audit_sha256': sha(prior / 'independent_audit.json'),
                'reference_results_sha256': sha(prior / 'results.json'),
                'output_tolerance': {'rtol': 2e-12, 'atol': 2e-12},
                'gradient_tolerance': {'rtol': 2e-6, 'atol': 2e-7},
                'binomial_trials': 5000, 'binomial_samples': [1, 16, 1024],
                'moment_test_standard_errors': 6,
                'scope': 'Same-public-draw ACT12 affine operator parity, declared first-order STE algebra, nominal IID binomial moments. Not full-model execution, unbiased sampler gradients, finite precision native marginal identity or trained language quality.'}
    with (out / 'protocol.json').open('x') as f:
        json.dump(protocol, f, indent=2)
    rng = np.random.default_rng(202610320)
    records = []
    for index, rec in enumerate(reference['receipts']):
        program_rec = next(v for v in reference['programs'] if v['kind'] == rec['kind'] and v['variant'] == rec['variant'])
        with np.load(program_rec['program']) as program:
            shape = tuple(program['shape'])
            encoder = OperatorProbabilityTensor(torch.zeros(shape, dtype=torch.bfloat16))
            with torch.no_grad():
                encoder.theta.copy_(torch.from_numpy(program['codes'].astype(np.float32) * np.float32(8 / 4095) - np.float32(4)))
                encoder.lower.copy_(torch.from_numpy(program['lower']))
                encoder.span.copy_(torch.from_numpy(program['span']))
            codes = program['codes'].copy()
        payload = np.load(rec['input'])
        canonical_counts = None
        if rec['mode'] == 'native':
            worker_counts = np.empty(payload['codes'].shape, dtype=np.int64)
            for tile, start in enumerate(range(0, worker_counts.shape[1], 64)):
                raw = np.load(prior / 'raw_states' / f"call{rec['call']}_tile{tile}.npz")
                width = min(64, worker_counts.shape[1] - start)
                worker_counts[:, start:start + width] = raw['states'].sum(0, dtype=np.int64).reshape(worker_counts.shape[0], 64)[:, :width]
            canonical_counts = (worker_counts.T if rec['kind'] in ['linear', 'conv'] else worker_counts).reshape(shape).copy()
        inputs = torch.from_numpy(payload['x'].copy()).requires_grad_()
        weight = encoder.weights(samples=rec['samples'] if canonical_counts is not None else 0,
                                 counts=None if canonical_counts is None else torch.from_numpy(canonical_counts))
        y = execute(rec['kind'], inputs, weight)
        np.testing.assert_allclose(y.detach().numpy(), np.load(rec['output']), rtol=2e-12, atol=2e-12)
        upstream = torch.from_numpy(rng.standard_normal(y.shape))
        (y * upstream).sum().backward()
        assert torch.isfinite(encoder.theta.grad).all() and torch.isfinite(inputs.grad).all()
        _, integers, scale = quantize_act12(inputs)
        path = out / f'case{index:02d}.npz'
        np.savez_compressed(path, x=inputs.detach().numpy(), codes=codes,
            lower=encoder.lower.numpy(), span=encoder.span.numpy(),
            counts=np.empty(0, dtype=np.int64) if canonical_counts is None else canonical_counts,
            integers=integers.numpy(), scale=np.asarray(scale), weights=weight.detach().numpy(),
            output=y.detach().numpy(), upstream=upstream.numpy(), theta_gradient=encoder.theta.grad.numpy(), input_gradient=inputs.grad.numpy())
        records.append({'path': str(path), 'file_sha256': sha(path), 'kind': rec['kind'],
                        'variant': rec['variant'], 'mode': rec['mode'], 'samples': rec['samples'],
                        'reference_output': rec['output'], 'reference_output_sha256': sha(Path(rec['output']))})
    moment_encoder = OperatorProbabilityTensor(torch.tensor([0., .01, .1, .5, .9, .99, 1.], dtype=torch.bfloat16))
    codes, fields = moment_encoder.commands()
    probabilities = torch.sigmoid(2 * fields.double()).detach()
    generator = torch.Generator(device='cpu').manual_seed(202610321)
    law = {'codes': codes.numpy(), 'probabilities': probabilities.numpy()}
    for samples in protocol['binomial_samples']:
        p = probabilities.expand(protocol['binomial_trials'], -1)
        law[f'counts_{samples}'] = torch.binomial(torch.full_like(p, samples), p, generator=generator).numpy().astype(np.int64)
    law_path = out / 'binomial_law.npz'
    np.savez_compressed(law_path, **law)
    with (out / 'results.json').open('x') as f:
        json.dump({'status': 'all34_ACT12_operator_public_draw_outputs_and_gradients_saved',
                   'cases': records, 'law_path': str(law_path), 'law_sha256': sha(law_path)}, f, indent=2)
    print('All34 ACT12 operator cases passed; saved gradients and binomial moment data')


if __name__ == '__main__':
    main()
