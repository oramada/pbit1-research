"""Preselected batch-only diagnostic; never applies the full-cohort quality gate."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(16 * 1024 * 1024), b''): h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--batch', type=int, required=True)
    args = parser.parse_args(); torch.set_num_threads(2)
    root = Path('results/thrml_spw/lfm3_1p7b')
    shared = root / 'counted_mean_validation_protocol.json'
    p = json.loads(shared.read_text()); assert args.seed in p['seeds'] and 0 <= args.batch < 4
    case = root / 'counted_mean_validation/mean' / str(args.seed)
    path = case / f'batch{args.batch}.npz'
    run = json.loads((case / 'protocol.json').read_text())
    assert run['shared_protocol_sha256'] == sha(shared) and run['arm'] == 'mean' and run['seed'] == args.seed
    for name, digest in p['sources_sha256'].items(): assert sha(Path(__file__).with_name(name)) == digest
    package = root / 'adaptation_pilot/mean/program_package.json'
    assert run['package_sha256'] == sha(package)
    expected = np.load(root / 'adaptation_data/tokens.npz')['validation_tokens'][8 * args.batch:8 * (args.batch + 1)]
    out = root / 'counted_mean_diagnostics' / f'seed{args.seed}_batch{args.batch}'; out.mkdir(parents=True, exist_ok=False)
    protocol = {'source_sha256': sha(Path(__file__)), 'case_protocol_sha256': sha(case / 'protocol.json'),
        'shared_protocol_sha256': sha(shared), 'batch_output_sha256': sha(path),
        'weight_package_sha256': sha(package), 'seed': args.seed, 'batch': args.batch,
        'selection': 'Fixed ordinal seed350/batch0 chosen as first completed batch before losses were inspected; no favorable-score selection.',
        'scope': 'Interim diagnostic only. Never stops or changes the full five-seed/2048target protocol, never evaluates its quality gate, and does not stand in for finalcase coverage/program/resource audit.'}
    with (out / 'protocol.json').open('x') as f: json.dump(protocol, f, indent=2)
    values, metrics = {}, {}
    with np.load(path) as a:
        np.testing.assert_array_equal(a['input_ids'], expected)
        for stage in ['BF16', 'same_fit_BF16', 'nominal_controls', 'native']:
            y = a[stage]; assert y.dtype == np.float32 and y.shape == (8, 65, 128000) and np.isfinite(y).all()
            windows = []
            for i in range(8):
                z = y[i, :-1].astype(np.float64); targets = expected[i, 1:]
                highest = z.max(-1)
                numpy_loss = highest + np.log(np.exp(z - highest[:, None]).sum(-1)) - z[np.arange(64), targets]
                torch_loss = F.cross_entropy(torch.from_numpy(z), torch.from_numpy(targets.copy()), reduction='none').numpy()
                np.testing.assert_allclose(numpy_loss, torch_loss, rtol=1e-12, atol=1e-12)
                windows.append(torch_loss)
            loss = np.stack(windows); values[stage] = loss
            metrics[stage] = {'NLL_FP64': float(loss.mean()), 'subset_perplexity': float(np.exp(loss.mean()))}
    gaps = {stage: {'mean_paired_native_minus_reference_NLL': float(np.mean(values['native'] - values[stage])),
                     'exp_mean_gap_PPL_ratio': float(np.exp(np.mean(values['native'] - values[stage])))}
            for stage in ['BF16', 'same_fit_BF16', 'nominal_controls']}
    np.savez(out / 'token_losses.npz', **values)
    report = {'status': 'first_batch_losses_crosschecked_by_NumPy_and_Torch_FP64',
        'protocol_sha256': sha(out / 'protocol.json'), 'seed': args.seed, 'batch': args.batch, 'targets': 512,
        'MC_seed_clusters': 1, 'gate_evaluated': False, 'confidence_interval': None,
        'metrics': metrics, 'paired_native_gaps': gaps, 'token_losses_sha256': sha(out / 'token_losses.npz'),
        'scope': protocol['scope'] + ' Shared stochastic weights across positions; tokens are not512independent MC seeds. UnadaptedBF16 comparison includes extra-training benefit; same-fitBF16 isolates representation/deployment error. Not fullcohort performance, BF16 equivalence or unseen-textgeneralization.'}
    with (out / 'metrics.json').open('x') as f: json.dump(report, f, indent=2)
    print(json.dumps(report))


if __name__ == '__main__':
    main()
