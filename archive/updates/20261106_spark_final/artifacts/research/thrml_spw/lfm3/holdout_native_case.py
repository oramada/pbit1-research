"""Prospective complete reserved evaluation; cannot run without frozen gates.

Not adopted by an active experiment. Requires a separately frozen complete
execution/statistics protocol and qualifying five-seed validation evidence.
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
from .holdout_batch_layout import batch_plan, assemble_losses
from .holdout_frontend import prepare, restore
from .holdout_treatment_selection_v2 import select_treatment
from .lossless_logits import write_logits
from .program_export import load_program


STAGES = ['BF16', 'same_fit_BF16', 'nominal_controls', 'native']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(16 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def validate_contract(protocol, selection, plan, seed):
    """Pure contract checks; never open tokens or authorize from partial results."""
    if protocol.get('status') != 'complete_execution_and_statistics_protocol_frozen_before_scoring':
        raise ValueError('Complete frozen execution/statistics protocol required')
    if selection.get('status') != 'eligible_after_full_execution_protocol_freeze':
        raise ValueError('Complete qualifying validation cohort required')
    if selection.get('pbits_per_parameter') != 1 or selection.get('arm') != 'mean':
        raise ValueError('Expected the selected single-p-bit mean treatment')
    if protocol.get('samples') != selection.get('samples') or protocol.get('samples') not in (1024, 4096, 8192):
        raise ValueError('Treatment differs from independently selected budget')
    seeds = protocol.get('seeds', [])
    if len(seeds) != 5 or any(type(v) is not int or not 0 <= v < 2**32 for v in seeds) or len(set(seeds)) != 5 or seed not in seeds:
        raise ValueError('Five distinct fixed MC keys required')
    if protocol.get('stages') != STAGES or protocol.get('batch_plan') != plan:
        raise ValueError('Complete fixed controls and batch geometry required')
    if protocol.get('targets') != 10240 or protocol.get('articles') != 40:
        raise ValueError('All reserved targets and articles required')
    if protocol.get('field_bits') != 12 or protocol.get('activation_bits') != 12:
        raise ValueError('The validated finite command/activation precision is fixed')
    if not protocol.get('statistics_protocol_sha256') or not protocol.get('independent_case_verifier_sha256'):
        raise ValueError('Frozen statistical analysis and independent case verifier required')
    return True


def token_nll(logits, ids):
    """Bound float64 temporary memory to sixteen positions, keeping all targets."""
    if logits.ndim != 3 or ids.ndim != 2 or logits.shape[:2] != ids.shape or logits.dtype != np.float32:
        raise ValueError('Logits and input geometry disagree')
    if ids.dtype.kind not in 'iu' or ids.size == 0 or ids.shape[1] < 2 or ids.min() < 0 or ids.max() >= logits.shape[2]:
        raise ValueError('Nonempty valid integer input/target IDs required')
    losses = np.empty((ids.shape[0], ids.shape[1] - 1), dtype=np.float64)
    for window in range(ids.shape[0]):
        for start in range(0, ids.shape[1] - 1, 16):
            stop = min(start + 16, ids.shape[1] - 1)
            z = logits[window, start:stop].astype(np.float64)
            maximum = z.max(-1)
            targets = ids[window, start + 1:stop + 1]
            losses[window, start:stop] = maximum + np.log(np.exp(z - maximum[:, None]).sum(-1)) - z[np.arange(stop-start), targets]
    if not np.isfinite(losses).all():
        raise ValueError('Nonfinite token loss')
    return losses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--protocol', type=Path, default=Path('results/thrml_spw/lfm3_1p7b/holdout_native_protocol.json'))
    options = parser.parse_args()
    root = Path('results/thrml_spw/lfm3_1p7b')
    protocol = json.loads(options.protocol.read_text())
    # All inputs and executed source bytes must be bound in the final protocol.
    required_sources = {Path(__file__).name, 'holdout_batch_layout.py', 'holdout_frontend.py',
        'holdout_treatment_selection_v2.py', 'holdout_treatment_selection.py', 'count_budget_continuation.py',
        'lossless_logits.py', 'program_export.py', 'counted_programmed_worker.py', 'counted_bank.py'}
    if not required_sources <= set(protocol['sources_sha256']):
        raise ValueError('Incomplete executed-source binding')
    for name, digest in protocol['sources_sha256'].items():
        if sha(Path(__file__).with_name(name)) != digest:
            raise ValueError('Executed source changed: ' + name)
    required_inputs = {'holdout_data/manifest.json', 'holdout_data/tokens.npz',
        'holdout_data/independent_audit.json', 'holdout_treatment_selection_v2_policy.json',
        'adaptation_pilot/mean/program_package.json', 'adaptation_pilot/mean/independent_audit.json',
        'adaptation_pilot/mean/module_aliases.json', 'holdout_native_statistics_protocol.json',
        'counted_mean_validation_protocol.json', 'counted_mean_t4096_protocol.json', 'counted_mean_t8192_protocol.json'}
    for directory in ['counted_mean_validation', 'counted_mean_t4096', 'counted_mean_t8192']:
        required_inputs.update({directory + '/statistics.json', directory + '/statistics_audit.json'})
    if not required_inputs <= set(protocol['inputs_sha256']):
        raise ValueError('Incomplete immutable input binding')
    if protocol['inputs_sha256']['holdout_native_statistics_protocol.json'] != protocol['statistics_protocol_sha256']:
        raise ValueError('Statistical protocol hash disagrees')
    if protocol['sources_sha256'].get('verify_holdout_native_case.py') != protocol['independent_case_verifier_sha256']:
        raise ValueError('Independent case verifier is not source bound')
    for name, digest in protocol['inputs_sha256'].items():
        if sha(root / name) != digest:
            raise ValueError('Frozen input changed: ' + name)
    for name, digest in protocol['auxiliary_inputs_sha256'].items():
        if sha(name) != digest:
            raise ValueError('Frozen auxiliary input changed: ' + name)
    checkpoint = 'data/thrml_spw/lfm3_1p7b/checkpoint/'
    if not {checkpoint + n for n in ['model.safetensors', 'config.json', 'modeling_lfm3.py',
            'configuration_lfm3.py', 'lfm3_conv.py']} <= set(protocol['auxiliary_inputs_sha256']):
        raise ValueError('Checkpoint weights and executed model sources must be bound')
    policy_path = root / 'holdout_treatment_selection_v2_policy.json'
    if sha(policy_path) != protocol['eligibility_policy_sha256']:
        raise ValueError('Eligibility policy changed')
    policy = json.loads(policy_path.read_text())
    bundles = []
    for directory in ['counted_mean_validation', 'counted_mean_t4096', 'counted_mean_t8192']:
        stat, audit = root / directory / 'statistics.json', root / directory / 'statistics_audit.json'
        bundles.append(None if not stat.exists() or not audit.exists() else {
            'statistics': json.loads(stat.read_text()), 'audit': json.loads(audit.read_text()),
            'statistics_sha256': sha(stat)})
    selection = select_treatment(*bundles, policy)
    previous_seeds = set()
    for name in ['counted_mean_validation_protocol.json', 'counted_mean_t4096_protocol.json', 'counted_mean_t8192_protocol.json']:
        previous_seeds.update(json.loads((root / name).read_text())['seeds'])
    if previous_seeds.intersection(protocol['seeds']):
        raise ValueError('Fresh MC keys distinct from validation required')
    manifest_path = root / 'holdout_data/manifest.json'
    if sha(manifest_path) != policy['holdout_manifest_sha256']:
        raise ValueError('Reserved manifest changed')
    manifest = json.loads(manifest_path.read_text())
    plan = batch_plan(manifest)
    validate_contract(protocol, selection, plan, options.seed)
    # No reserved token array is loaded before all the checks above succeed.
    with np.load(root / 'holdout_data/tokens.npz', allow_pickle=False) as arrays:
        inputs = {label: arrays[label + '_tokens'].copy() for label in ['short', 'long']}
    for label, expected in [('short', (96, 65)), ('long', (16, 257))]:
        if inputs[label].shape != expected:
            raise ValueError('Reserved token shape changed')
    import torch
    from transformers import AutoModelForCausalLM
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required for the full model')
    out = root / 'holdout_native' / str(options.seed)
    out.mkdir(parents=True, exist_ok=False)
    package_dir = root / 'adaptation_pilot/mean'
    package = json.loads((package_dir / 'program_package.json').read_text())
    if sha(package_dir / 'program_package.json') != policy['adapted_mean_package_sha256']:
        raise ValueError('Adapted point changed')
    aliases = json.loads((package_dir / 'module_aliases.json').read_text())
    records = {v['name']: v for v in package['programs']}
    programs = {}
    for name, rec in records.items():
        if sha(rec['program']) != rec['program_sha256']:
            raise ValueError('Program changed')
        programs[name] = load_program(rec['program'])
    assert len(programs) == 286 and len(aliases) == 287
    assert sum(p['codes'].size for p in programs.values()) == 1728801536
    run_protocol = {'seed': options.seed, 'samples': protocol['samples'], 'selection': selection,
        'shared_protocol_sha256': sha(options.protocol), 'package_sha256': sha(package_dir / 'program_package.json')}
    with (out / 'protocol.json').open('x') as f:
        json.dump(run_protocol, f, indent=2)
    begin = time.monotonic()
    def progress(stage, **extra):
        temp = out / 'progress.tmp'
        temp.write_text(json.dumps({'stage': stage, 'elapsed_seconds': time.monotonic()-begin, **extra}) + '\n')
        temp.replace(out / 'progress.json')
    progress('loading_BF16_backbone')
    model = AutoModelForCausalLM.from_pretrained('data/thrml_spw/lfm3_1p7b/checkpoint',
        trust_remote_code=True, local_files_only=True, dtype=torch.bfloat16, attn_implementation='eager').cuda().eval()
    sys.modules[type(model.model.layers[0].conv).__module__].can_use_lfm3_conv = lambda *args: False
    targets = [(name, module) for name, module in model.named_modules() if module._parameters.get('weight') is not None]
    assert {name for name, _ in targets} == set(aliases)
    assert sum(p.numel() for p in model.parameters()) == 1728801536
    original = {aliases[name]: module.weight.detach() for name, module in targets}
    same_fit, weight_hashes = {}, {}
    for name, p in programs.items():
        fields = p['codes'].astype(np.float32) * np.float32(8/4095) - np.float32(4)
        probability = 1/(1+np.exp(-2*fields.astype(np.float64)))
        rows = fields.shape[0] if fields.ndim > 1 else 1
        expected = (p['lower'].astype(np.float64) + p['span'].astype(np.float64) * probability.reshape(rows, -1, int(p['group_size']))).reshape(fields.shape)
        w = torch.from_numpy(expected).to(torch.bfloat16)
        weight_hashes[name] = hashlib.sha256(w.view(torch.uint16).numpy().tobytes()).hexdigest()
        same_fit[name] = w.cuda()
    del fields, probability, expected, w
    mode, batch, module_calls = 'BF16', 0, []
    all_calls, handles, outputs, metrics = [], [], [], []
    worker = None
    log = (out / 'worker.log').open('x')
    def before_hook(name):
        def hook(module, args):
            if mode == 'same_fit_BF16':
                assert module.weight.data_ptr() == original[aliases[name]].data_ptr()
                module.weight.data = same_fit[aliases[name]]
        return hook
    def after_hook(name, module):
        kind = ('embedding' if isinstance(module, torch.nn.Embedding) else 'linear' if isinstance(module, torch.nn.Linear)
                else 'conv' if isinstance(module, torch.nn.Conv1d) else 'elementwise')
        p = programs[aliases[name]]
        assert tuple(module.weight.shape) == tuple(p['codes'].shape) and getattr(module, 'bias', None) is None
        def hook(mod, args, deterministic):
            if mode in ['BF16', 'same_fit_BF16']:
                if mode == 'same_fit_BF16':
                    assert module.weight.data_ptr() == same_fit[aliases[name]].data_ptr()
                    module.weight.data = original[aliases[name]]
                module_calls.append(name)
                return deterministic
            value = args[0]
            if kind == 'elementwise':
                assert hasattr(module, 'variance_epsilon')
                value = (value.float() * torch.rsqrt(value.float().square().mean(-1, keepdim=True) + module.variance_epsilon)).to(value.dtype)
            x = value.detach().cpu().numpy() if kind == 'embedding' else value.detach().float().cpu().numpy()
            xx, layout, state = prepare(kind, x, p)
            ip, op = out / 'runtime_input.npz', out / 'runtime_output.npy'
            np.savez(ip, x=xx, **layout)
            call = 287 + batch * 287 + len(module_calls)
            req = {'input': str(ip), 'output': str(op), 'kind': kind, 'samples': protocol['samples'],
                'field_bits': 12, 'activation_bits': 12, 'key': options.seed, 'call': call,
                'mode': 'native' if mode == 'native' else 'mean'}
            worker.stdin.write(json.dumps(req)+'\n'); worker.stdin.flush()
            line = worker.stdout.readline()
            if not line:
                raise RuntimeError('Official THRML worker terminated; preserve logs')
            receipt = json.loads(line)
            assert receipt['ok'] and receipt['finite'] and receipt['program_codes_used_directly'] and not receipt['reference_weights_supplied']
            y = restore(np.load(op), state)
            module_calls.append(name)
            all_calls.append({'batch': batch, 'cohort': plan[batch]['cohort'], 'stage': mode,
                'module': name, 'canonical_program': aliases[name], 'kind': kind,
                'program_sha256': records[aliases[name]]['program_sha256'],
                'embedding_rows': state['rows'].tolist() if kind == 'embedding' else None, **receipt})
            progress(mode, batch=batch, operator=len(module_calls), last_module=name)
            return torch.from_numpy(y.copy()).to(device=deterministic.device, dtype=deterministic.dtype)
        return hook
    try:
        for name, module in targets:
            handles.extend([module.register_forward_pre_hook(before_hook(name)), module.register_forward_hook(after_hook(name, module))])
        worker = subprocess.Popen([sys.executable, '-m', 'research.thrml_spw.gpu_environment', '--',
            '.venv-thrml-gpu/bin/python', '-m', 'research.thrml_spw.lfm3.counted_programmed_worker'],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True, bufsize=1)
        ready = json.loads(worker.stdout.readline())
        assert ready['ready'] and any('cuda' in v.lower() or 'gpu' in v.lower() for v in ready['devices'])
        for batch, spec in enumerate(plan):
            ids_array = inputs[spec['cohort']][spec['window_start']:spec['window_stop']]
            ids = torch.from_numpy(ids_array).cuda()
            for mode in STAGES:
                progress(mode, batch=batch, operator=0)
                module_calls.clear()
                with torch.inference_mode():
                    logits = model(input_ids=ids, use_cache=False).logits.float().cpu().numpy()
                assert Counter(module_calls) == Counter(aliases.keys())
                assert all(module.weight.data_ptr() == original[aliases[name]].data_ptr() for name, module in targets)
                assert logits.shape == (*spec['input_shape'], 128000) and np.isfinite(logits).all()
                path = out / f'batch{batch}_{mode}.npz'
                codec = write_logits(path, logits)
                losses = token_nll(logits, ids_array)
                outputs.append({'batch': batch, 'stage': mode, 'path': str(path), 'sha256': sha(path), **codec})
                metrics.append({'batch_index': batch, 'stage': mode, 'mean_NLL_FP64': float(losses.mean()), 'token_NLL': losses.tolist()})
                del logits
        for stage in STAGES:
            assemble_losses(plan, [{'batch_index': v['batch_index'], 'token_NLL': v['token_NLL']} for v in metrics if v['stage'] == stage])
        result = {'status': 'full_reserved_native_case_produced_pending_independent_audit',
            'seed': options.seed, 'samples': protocol['samples'], 'pbits_per_parameter': 1,
            'unique_programs': 286, 'weighted_modules': 287, 'unique_parameter_coordinates': 1728801536,
            'targets': 10240, 'articles': 40, 'outputs': outputs, 'metrics': metrics,
            'operator_calls': all_calls, 'same_fit_BF16_weight_hashes': weight_hashes,
            'worker_ready': ready, 'shared_protocol_sha256': sha(options.protocol),
            'elapsed_seconds': time.monotonic()-begin,
            'scope': 'Complete fixed teacher-forced short/long article holdout, pending independent audit. Existing deterministic operators also run before hooks; wall time is simulator overhead, not hardware latency. Full raw float32 observations retained losslessly. No physical device, energy, numerical BF16 equivalence or autoregressive-quality claim.'}
        with (out / 'results.json').open('x') as f:
            json.dump(result, f, indent=2)
        progress('complete_pending_independent_audit')
    except Exception as error:
        with (out / 'failure.json').open('x') as f:
            json.dump({'type': type(error).__name__, 'message': str(error), 'status': 'failed_preserved_no_restart'}, f, indent=2)
        raise
    finally:
        for handle in handles:
            handle.remove()
        if worker is not None:
            if worker.poll() is None:
                try:
                    worker.stdin.write('{"stop":true}\n'); worker.stdin.flush(); worker.stdin.close()
                except BrokenPipeError:
                    pass
            worker.wait(timeout=60)
        log.close()


if __name__ == '__main__':
    main()
