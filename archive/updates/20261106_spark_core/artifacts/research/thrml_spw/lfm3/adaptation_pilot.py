"""Prospective all-parameter ACT12 LFM3 probability adaptation pilot.

Mean and IID binomial training are differentiable Torch surrogates. Every
trained finite program must subsequently be evaluated in official THRML.
Preserve every update gradient and final resumable FP32 Adam/control state.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import resource
import shutil
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from .probability_operators import install
from .program_export import export_program


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda: f.read(16 * 1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def array_sha(array):
    return hashlib.sha256(array.tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=['mean', 'binomial', 'baseline'], required=True)
    options = parser.parse_args()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    root = Path('results/thrml_spw/lfm3_1p7b')
    shared_path = root / 'adaptation_pilot_protocol.json'
    protocol = json.loads(shared_path.read_text())
    assert protocol['status'] == 'prospectively_frozen_before_training'
    for name, digest in protocol['sources_sha256'].items():
        assert sha(Path(__file__).with_name(name)) == digest
    for path, digest in protocol['prerequisites_sha256'].items():
        assert sha(root / path) == digest
    assert protocol['steps'] == 16 and protocol['train_window_length'] == 65
    assert protocol['samples'] == 1024 and protocol['learning_rate'] == .0001
    assert shutil.disk_usage(root).free > 200 * 1024 ** 3
    out = root / 'adaptation_pilot' / options.arm
    out.mkdir(parents=True, exist_ok=True)
    # Exclusive protocol creation is also the no-restart guard.
    with (out / 'protocol.json').open('x') as f:
        json.dump({'shared_protocol_sha256': sha(shared_path), 'arm': options.arm,
                   'source_sha256': sha(Path(__file__))}, f, indent=2)
    for folder in ['gradients', 'programs', 'checkpoints', 'validation']:
        (out / folder).mkdir(exist_ok=True)
    begin = time.monotonic()
    def progress(stage, **values):
        (out / 'progress.json').write_text(json.dumps({'stage': stage, 'elapsed_seconds': time.monotonic() - begin, **values}) + '\n')
    progress('loading_checkpoint')
    model = AutoModelForCausalLM.from_pretrained('data/thrml_spw/lfm3_1p7b/checkpoint',
        trust_remote_code=True, local_files_only=True, dtype=torch.bfloat16, attn_implementation='eager').cpu().eval()
    sys.modules[type(model.model.layers[0].conv).__module__].can_use_lfm3_conv = lambda *args: False
    data = np.load(root / 'adaptation_data/tokens.npz')
    train = data['train_tokens'][:protocol['steps'], :protocol['train_window_length']]
    validation = data['validation_tokens']
    assert train.shape == (16, 65) and validation.shape == (32, 65)
    bf16_validation = None
    if options.arm == 'baseline':
        losses, batches = [], []
        with torch.inference_mode():
            for batch, start in enumerate(range(0, len(validation), protocol['validation_batch_size'])):
                progress('BF16_validation', batch=batch)
                ids = torch.from_numpy(validation[start:start + protocol['validation_batch_size']])
                logits = model(input_ids=ids, use_cache=False).logits.float()
                nll = F.cross_entropy(logits[:, :-1].double().reshape(-1, logits.shape[-1]),
                    ids[:, 1:].reshape(-1), reduction='none').reshape(len(ids), -1).numpy()
                assert np.isfinite(nll).all()
                path = out / 'validation' / f'BF16_batch{batch}.npz'
                np.savez(path, input_ids=ids.numpy(), logits=logits.numpy(), token_NLL=nll)
                losses.append(nll); batches.append({'path': str(path), 'sha256': sha(path)})
        nll = np.concatenate(losses)
        bf16_validation = {'mean_NLL_FP64': float(nll.mean()), 'targets': nll.size,
                           'subset_perplexity': float(np.exp(nll.mean())), 'batches': batches}
        with (out / 'validation' / 'BF16.json').open('x') as f:
            json.dump(bf16_validation, f, indent=2)
    progress('installing_probability_operators')
    state, encoders, aliases = install(model, seed=protocol['training_seed'])
    assert len(encoders) == 286 and len(aliases) == 287
    assert sum(e.theta.numel() for e in encoders.values()) == 1728801536
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 1728801536
    assert model.model.embed_tokens.spw_encoder is model.lm_head.spw_encoder
    (out / 'module_aliases.json').write_text(json.dumps(aliases, indent=2) + '\n')
    initial_fingerprints = {name: e.fingerprints() for name, e in encoders.items()}
    with (out / 'initial_fingerprints.json').open('x') as f:
        json.dump(initial_fingerprints, f, indent=2)

    def forward(ids):
        state['calls'].clear()
        logits = model(input_ids=ids, use_cache=False).logits.float()
        assert Counter(state['calls']) == Counter(aliases.keys())
        assert torch.isfinite(logits).all()
        return logits

    def evaluate(label, samples, seed):
        state['samples'] = samples
        state['generator'].manual_seed(seed)
        losses = []
        records = []
        with torch.inference_mode():
            for batch, start in enumerate(range(0, len(validation), protocol['validation_batch_size'])):
                progress('validation', label=label, batch=batch)
                ids = torch.from_numpy(validation[start:start + protocol['validation_batch_size']])
                logits = forward(ids)
                token_loss = F.cross_entropy(logits[:, :-1].double().reshape(-1, logits.shape[-1]),
                    ids[:, 1:].reshape(-1), reduction='none').reshape(len(ids), -1).numpy()
                assert np.isfinite(token_loss).all()
                path = out / 'validation' / f'{label}_batch{batch}.npz'
                np.savez(path, input_ids=ids.numpy(), logits=logits.numpy(), token_NLL=token_loss)
                records.append({'path': str(path), 'sha256': sha(path), 'operator_modules': len(state['calls'])})
                losses.append(token_loss)
        nll = np.concatenate(losses)
        result = {'label': label, 'samples': samples, 'seed': seed, 'targets': nll.size,
                  'mean_NLL_FP64': float(nll.mean()), 'subset_perplexity': float(np.exp(nll.mean())),
                  'batches': records}
        with (out / 'validation' / (label + '.json')).open('x') as f:
            json.dump(result, f, indent=2)
        return result

    try:
        before_mean = evaluate('before_mean', 0, protocol['validation_seeds'][0])
        training_records = []
        optimizer = None
        if options.arm != 'baseline':
            optimizer = torch.optim.Adam([e.theta for e in encoders.values()],
                lr=protocol['learning_rate'], betas=(.9, .999), eps=1e-8, weight_decay=0, foreach=False)
            state['samples'] = protocol['samples'] if options.arm == 'binomial' else 0
            state['generator'].manual_seed(protocol['training_seed'])
            for step, window in enumerate(train, 1):
                progress('training_forward_backward', step=step, samples=state['samples'])
                ids = torch.from_numpy(window[None])
                optimizer.zero_grad(set_to_none=True)
                logits = forward(ids)
                loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1))
                assert torch.isfinite(loss)
                loss.backward()
                folder = out / 'gradients' / f'step{step:03d}'
                folder.mkdir()
                gradients = []
                squared_norm = 0.
                for index, (name, e) in enumerate(encoders.items()):
                    gradient = e.theta.grad
                    assert gradient is not None and gradient.dtype == torch.float32 and torch.isfinite(gradient).all()
                    path = folder / f'g{index:03d}.npy'
                    with path.open('xb') as f:
                        np.save(f, gradient.numpy())
                    squared_norm += float(gradient.double().square().sum())
                    gradients.append({'name': name, 'path': str(path), 'sha256': sha(path),
                        'array_sha256': array_sha(gradient.numpy()), 'numel': gradient.numel(),
                        'nonzero': int(torch.count_nonzero(gradient))})
                assert sum(v['numel'] for v in gradients) == 1728801536
                progress('optimizer_update', step=step)
                optimizer.step()
                for e in encoders.values():
                    e.project()
                    assert torch.isfinite(e.theta).all()
                training_records.append({'step': step, 'loss_FP32': float(loss.detach()),
                    'train_window_index': step - 1, 'targets': 64, 'gradient_L2': squared_norm ** .5,
                    'gradients': gradients})
                with (out / f'step{step:03d}.json').open('x') as f:
                    json.dump(training_records[-1], f, indent=2)
                del logits, loss
            optimizer.zero_grad(set_to_none=True)
        records = []
        for index, (name, encoder) in enumerate(encoders.items()):
            progress('exporting_final_program_and_checkpoint', completed=index, last_parameter=name)
            program = out / 'programs' / f'p{index:03d}.npz'
            export_program(encoder, program)
            checkpoint = out / 'checkpoints' / f'p{index:03d}.npz'
            arrays = {'theta': encoder.theta.detach().numpy()}
            if optimizer is not None:
                s = optimizer.state[encoder.theta]
                assert float(s['step']) == protocol['steps']
                arrays.update(exp_avg=s['exp_avg'].numpy(), exp_avg_sq=s['exp_avg_sq'].numpy(), step=s['step'].numpy())
            with checkpoint.open('xb') as f:
                np.savez(f, **arrays)
            records.append({'name': name, 'shape': list(encoder.original_shape), 'numel': encoder.theta.numel(),
                'program': str(program), 'program_sha256': sha(program), 'checkpoint': str(checkpoint),
                'checkpoint_sha256': sha(checkpoint), 'state_array_hashes': {k: array_sha(v) for k, v in arrays.items()},
                **encoder.fingerprints()})
        rng_path = out / 'training_rng_state.npy'
        np.save(rng_path, state['generator'].get_state().numpy())
        # Persist the program package before long validation sampling begins.
        package = {'programs': records, 'training_steps': len(training_records),
                   'training_rng_state_sha256': sha(rng_path), 'module_aliases_sha256': sha(out / 'module_aliases.json'),
                   'initial_fingerprints_sha256': sha(out / 'initial_fingerprints.json')}
        with (out / 'program_package.json').open('x') as f:
            json.dump(package, f, indent=2)
        if optimizer is not None:
            del optimizer
        mean_after = evaluate('after_mean', 0, protocol['validation_seeds'][0])
        sampled = [evaluate(f'after_binomial_{seed}', protocol['samples'], seed) for seed in protocol['validation_seeds']]
        report = {'status': 'all_parameter_probability_adaptation_and_validation_surrogate_complete',
            'arm': options.arm, 'unique_controls': 1728801536, 'unique_encoders': 286, 'weighted_modules': 287,
            'BF16_validation': bf16_validation,
            'before_mean': before_mean, 'after_mean': mean_after, 'after_binomial': sampled,
            'training_steps': len(training_records), 'training_targets': 64 * len(training_records),
            'program_package_sha256': sha(out / 'program_package.json'),
            'elapsed_seconds': time.monotonic() - begin,
            'process_peak_RSS_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            'scope': protocol['scope']}
        with (out / 'results.json').open('x') as f:
            json.dump(report, f, indent=2)
        progress('complete')
        print(json.dumps({k: v for k, v in report.items() if k not in ['before_mean', 'after_mean', 'after_binomial']}), flush=True)
    except Exception as error:
        with (out / 'failure.json').open('x') as f:
            json.dump({'status': 'failed_preserved_no_restart', 'type': type(error).__name__,
                       'message': str(error), 'last_progress': json.loads((out / 'progress.json').read_text())}, f, indent=2)
        raise


if __name__ == '__main__':
    main()
