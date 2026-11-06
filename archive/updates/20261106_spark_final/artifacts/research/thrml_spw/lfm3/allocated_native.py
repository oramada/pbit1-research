"""Execute audited full LFM3 finite-command packages in official GPU THRML.

All parameters, direct exported commands, fixed metadata, ACT12 group integer
accumulation. Existing deterministic operators also run before output hooks;
reported wall time is simulator overhead, not an accelerator latency model.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch
from transformers import AutoModelForCausalLM
from .program_export import load_program, operator_layout


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda: f.read(16 * 1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=['baseline', 'mean', 'binomial'], required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--replay', action='store_true')
    options = parser.parse_args()
    assert options.arm == 'mean' and not options.replay
    torch.set_num_threads(4)
    root = Path('results/thrml_spw/lfm3_1p7b')
    shared = root / 'allocated_native_protocol.json'
    protocol = json.loads(shared.read_text())
    for name, digest in protocol['sources_sha256'].items():
        assert sha(Path(__file__).with_name(name)) == digest
    for name, digest in protocol['auxiliary_sources_sha256'].items():
        assert sha(Path(name)) == digest
    for name, digest in protocol['inputs_sha256'].items():
        assert sha(root / name) == digest
    assert sha(root / 'adaptation_pilot_protocol.json') == protocol['adaptation_protocol_sha256']
    allocation_path = root / 'layer_allocation_candidates/results.json'
    allocation = json.loads(allocation_path.read_text())
    assert allocation['upstream_program_package_sha256'] == sha(root / 'adaptation_pilot/mean/program_package.json')
    samples_by_module = allocation['selected_samples']
    replay_audit_path = root / 'adaptation_native_v2/initial_replay/independent_audit.json'
    replay_audit = json.loads(replay_audit_path.read_text())
    assert replay_audit['status'] == 'all_direct_program_coverage_losses_and_resource_receipts_verified'
    assert replay_audit['initial_replay_exact_stages'] == ['BF16', 'nominal_controls', 'native']
    assert json.loads((root / 'sample8192_preflight/independent_audit.json').read_text())['status'] == 'all_T8192_public_counts_high_magnitude_int32_readouts_and_sign_mirrors_verified'
    gpu_audit = root / 'program_export_gpu_preflight/independent_audit.json'
    assert json.loads(gpu_audit.read_text())['status'] == 'all34_direct_command_serialization_mapping_public_counters_and_readouts_verified'
    package_dir = root / 'adaptation_pilot' / options.arm
    audit = json.loads((package_dir / 'independent_audit.json').read_text())
    assert audit['status'] == 'all_coordinates_Adam_trajectory_programs_and_validation_losses_verified'
    package = json.loads((package_dir / 'program_package.json').read_text())
    if options.replay:
        assert options.arm == 'baseline' and options.seed == 202610300
        inputs = np.load(root / 'language_subset/tokens.npz')['tokens']
        out = root / 'allocated_native' / 'initial_replay'
    else:
        assert options.seed in protocol['seeds']
        inputs = np.load(root / 'adaptation_data/tokens.npz')['validation_tokens']
        out = root / 'allocated_native' / options.arm / str(options.seed)
    assert inputs.shape == ((8, 65) if options.replay else (32, 65))
    assert torch.cuda.is_available()
    out.mkdir(parents=True, exist_ok=True)
    run_protocol = {'arm': options.arm, 'seed': options.seed, 'initial_replay': options.replay,
        'shared_protocol_sha256': sha(shared), 'package_sha256': sha(package_dir / 'program_package.json'),
        'package_audit_sha256': sha(package_dir / 'independent_audit.json'),
        'GPU_preflight_audit_sha256': sha(gpu_audit)}
    run_protocol.update(allocation_results_sha256=sha(allocation_path),
                        initial_replay_audit_sha256=sha(replay_audit_path))
    with (out / 'protocol.json').open('x') as f:
        json.dump(run_protocol, f, indent=2)
    begin = time.monotonic()
    def progress(stage, **values):
        (out / 'progress.json').write_text(json.dumps({'stage': stage, 'elapsed_seconds': time.monotonic() - begin, **values}) + '\n')
    progress('loading_direct_programs')
    records = {v['name']: v for v in package['programs']}
    programs = {}
    for name, rec in records.items():
        path = Path(rec['program'])
        assert sha(path) == rec['program_sha256']
        programs[name] = load_program(path)
    aliases = json.loads((package_dir / 'module_aliases.json').read_text())
    assert len(programs) == 286 and len(aliases) == 287
    assert set(samples_by_module) == set(aliases)
    assert all(v in [16,32,64,128,256,512,1024,2048,4096,8192] for v in samples_by_module.values())
    assert sum(p['codes'].size for p in programs.values()) == 1728801536
    progress('loading_BF16_backbone')
    model = AutoModelForCausalLM.from_pretrained('data/thrml_spw/lfm3_1p7b/checkpoint',
        trust_remote_code=True, local_files_only=True, dtype=torch.bfloat16, attn_implementation='eager').cuda().eval()
    sys.modules[type(model.model.layers[0].conv).__module__].can_use_lfm3_conv = lambda *args: False
    targets = [(name, module) for name, module in model.named_modules() if module._parameters.get('weight') is not None]
    assert set(aliases) == {name for name, _ in targets}
    assert sum(p.numel() for p in model.parameters()) == 1728801536
    for name, module in targets:
        assert tuple(module.weight.shape) == tuple(programs[aliases[name]]['codes'].shape)
        assert getattr(module, 'bias', None) is None
    original_weights = {aliases[name]: module.weight.detach() for name, module in targets}
    progress('building_same_fit_BF16_control')
    same_fit_weights, same_fit_hashes = {}, {}
    for name, program in programs.items():
        fields = program['codes'].astype(np.float32) * np.float32(8 / 4095) - np.float32(4)
        probability = 1 / (1 + np.exp(-2 * fields.astype(np.float64)))
        rows = fields.shape[0] if fields.ndim > 1 else 1
        width = fields.size // rows
        expected = (program['lower'].astype(np.float64) + program['span'].astype(np.float64)
                    * probability.reshape(rows, -1, int(program['group_size']))).reshape(fields.shape)
        weight = torch.from_numpy(expected).to(torch.bfloat16)
        same_fit_hashes[name] = hashlib.sha256(weight.view(torch.uint16).numpy().tobytes()).hexdigest()
        same_fit_weights[name] = weight.cuda()
    del fields, probability, expected, weight
    mode, batch, module_calls = 'BF16', 0, []
    all_calls, handles = [], []
    worker = None
    stderr = (out / 'worker.log').open('x')

    def make_pre_hook(name):
        def before(module, args):
            if mode == 'same_fit_BF16':
                assert module.weight.data_ptr() == original_weights[aliases[name]].data_ptr()
                module.weight.data = same_fit_weights[aliases[name]]
        return before

    def make_hook(name, module):
        kind = ('embedding' if isinstance(module, torch.nn.Embedding) else 'linear' if isinstance(module, torch.nn.Linear)
                else 'conv' if isinstance(module, torch.nn.Conv1d) else 'elementwise')
        program = programs[aliases[name]]
        def hook(mod, args, original):
            if mode == 'BF16':
                module_calls.append(name)
                return original
            if mode == 'same_fit_BF16':
                assert module.weight.data_ptr() == same_fit_weights[aliases[name]].data_ptr()
                module.weight.data = original_weights[aliases[name]]
                module_calls.append(name)
                return original
            rows = None
            restore = None
            if kind == 'embedding':
                tokens = args[0].detach().cpu().numpy()
                rows, inverse = np.unique(tokens, return_inverse=True)
                layout = operator_layout(program, kind, rows=rows)
                xx = np.ones((1, 1, layout['codes'].size), dtype=np.float32)
                restore = (tokens.shape, (len(rows), module.weight.shape[1]), inverse)
            elif kind == 'conv':
                x = args[0].detach().float().cpu().numpy()
                windows = np.lib.stride_tricks.sliding_window_view(np.pad(x, ((0, 0), (0, 0), (6, 0))), 7, axis=2)
                xx = windows.transpose(0, 2, 3, 1).reshape(-1, 7, x.shape[1])
                layout = operator_layout(program, kind)
            elif kind == 'linear':
                x = args[0].detach().float().cpu().numpy()
                xx = x.reshape(-1, x.shape[-1])
                layout = operator_layout(program, kind)
            else:
                assert module.weight.ndim == 1 and hasattr(module, 'variance_epsilon')
                z = args[0]
                normalized = (z.float() * torch.rsqrt(z.float().square().mean(-1, keepdim=True) + module.variance_epsilon)).to(z.dtype)
                x = normalized.detach().float().cpu().numpy()
                xx = x.reshape(-1, 1, x.shape[-1])
                layout = operator_layout(program, kind)
            ip, op = out / 'runtime_input.npz', out / 'runtime_output.npy'
            np.savez(ip, x=xx, **layout)
            # Fixed offsets permit exact replay of the prior single-p-bit run
            # and common-random-number coupling across all three candidates.
            call = 287 + batch * 287 + len(module_calls)
            planned_samples = samples_by_module[name] if mode == 'native' else protocol['samples']
            request = {'input': str(ip), 'output': str(op), 'kind': kind, 'samples': planned_samples,
                'field_bits': 12, 'activation_bits': 12, 'key': options.seed, 'call': call,
                'mode': 'native' if mode == 'native' else 'mean'}
            worker.stdin.write(json.dumps(request) + '\n'); worker.stdin.flush()
            line = worker.stdout.readline()
            if not line:
                raise RuntimeError('Official worker terminated; preserve worker log')
            receipt = json.loads(line)
            assert receipt['ok'] and receipt['finite'] and receipt['program_codes_used_directly']
            assert not receipt['reference_weights_supplied']
            y = np.load(op)
            if kind == 'embedding':
                input_shape, selected_shape, inverse = restore
                y = y.reshape(selected_shape)[inverse].reshape(*input_shape, selected_shape[-1])
            elif kind == 'conv':
                y = np.pad(y.reshape(x.shape[0], x.shape[2], x.shape[1]).transpose(0, 2, 1), ((0, 0), (0, 0), (0, 6)))
            else:
                y = y.reshape(*x.shape[:-1], module.weight.shape[0])
            module_calls.append(name)
            all_calls.append({'batch': batch, 'stage': mode, 'module': name, 'kind': kind,
                'planned_samples': planned_samples,
                'canonical_program': aliases[name], 'program_sha256': records[aliases[name]]['program_sha256'],
                'embedding_rows': None if rows is None else rows.tolist(), **receipt})
            progress(mode, batch=batch, operator=len(module_calls), last_module=name)
            return torch.from_numpy(y.copy()).to(device=original.device, dtype=original.dtype)
        return hook

    metrics, outputs = [], []
    try:
        for name, module in targets:
            handles.append(module.register_forward_pre_hook(make_pre_hook(name)))
            handles.append(module.register_forward_hook(make_hook(name, module)))
        worker = subprocess.Popen([sys.executable, '-m', 'research.thrml_spw.gpu_environment', '--',
            '.venv-thrml-gpu/bin/python', '-m', 'research.thrml_spw.lfm3.programmed_worker'],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr, text=True, bufsize=1)
        ready = json.loads(worker.stdout.readline())
        assert ready['ready'] and any('cuda' in v.lower() or 'gpu' in v.lower() for v in ready['devices'])
        for batch, start in enumerate(range(0, len(inputs), 8)):
            ids = torch.from_numpy(inputs[start:start + 8]).cuda()
            raw = {'input_ids': ids.cpu().numpy()}
            for mode in ['BF16', 'same_fit_BF16', 'nominal_controls', 'native']:
                progress(mode, batch=batch, operator=0)
                module_calls.clear()
                with torch.inference_mode():
                    logits = model(input_ids=ids, use_cache=False).logits.float().cpu().numpy()
                assert Counter(module_calls) == Counter(aliases.keys())
                assert all(module.weight.data_ptr() == original_weights[aliases[name]].data_ptr() for name, module in targets)
                assert logits.shape == (8, 65, 128000) and np.isfinite(logits).all()
                raw[mode] = logits
                z = logits[:, :-1].astype(np.float64)
                mx = z.max(-1)
                nll = mx + np.log(np.exp(z - mx[..., None]).sum(-1)) - np.take_along_axis(z, raw['input_ids'][:, 1:, None], -1)[..., 0]
                metrics.append({'batch': batch, 'stage': mode, 'mean_NLL_FP64': float(nll.mean()),
                    'subset_perplexity': float(np.exp(nll.mean())), 'token_NLL': nll.tolist(),
                    'logit_MSE_vs_BF16': float(np.mean((logits.astype(np.float64) - raw['BF16']) ** 2))})
            path = out / f'batch{batch}.npz'
            with path.open('xb') as f:
                np.savez(f, **raw)
            outputs.append({'path': str(path), 'sha256': sha(path)})
        result = {'status': 'all_exported_parameter_programs_native_validation_complete',
            'arm': options.arm, 'seed': options.seed, 'initial_replay': options.replay,
            'unique_programs': 286, 'unique_parameter_coordinates': 1728801536, 'weighted_modules': 287,
            'targets': len(inputs) * 64, 'outputs': outputs, 'metrics': metrics, 'operator_calls': all_calls,
            'same_fit_BF16_weight_hashes': same_fit_hashes,
            'worker_ready': ready, 'package_sha256': run_protocol['package_sha256'],
            'allocation_results_sha256': sha(allocation_path),
            'elapsed_seconds': time.monotonic() - begin,
            'scope': protocol['scope']}
        with (out / 'results.json').open('x') as f:
            json.dump(result, f, indent=2)
        progress('complete')
    except Exception as error:
        with (out / 'failure.json').open('x') as f:
            json.dump({'status': 'failed_preserved_no_restart', 'type': type(error).__name__, 'message': str(error)}, f, indent=2)
        raise
    finally:
        for handle in handles:
            handle.remove()
        if worker is not None:
            if worker.poll() is None:
                worker.stdin.write('{"stop":true}\n'); worker.stdin.flush(); worker.stdin.close()
            worker.wait(timeout=60)
        stderr.close()


if __name__ == '__main__':
    main()
