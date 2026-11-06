"""Full-model first-order sampling sensitivity on unused training windows.

No updates or validation/test-based budget fitting. Preserve every module-use
FP64 weight gradient so the diagnostic variance coefficients can be audited.
"""
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
from .layer_allocation import instrument, variance_coefficient, allocate


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda: f.read(16 * 1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def array_sha(array):
    h = hashlib.sha256()
    if array.ndim == 1:
        h.update(array.tobytes())
    else:
        for start in range(0, len(array), 128):
            h.update(array[start:start + 128].tobytes())
    return h.hexdigest()


def main():
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    root = Path('results/thrml_spw/lfm3_1p7b')
    out = root / 'layer_calibration'
    out.mkdir(exist_ok=True)
    assert shutil.disk_usage(root).free > 100 * 1024 ** 3
    training = root / 'adaptation_pilot/mean'
    audit = json.loads((training / 'independent_audit.json').read_text())
    assert audit['status'] == 'all_coordinates_Adam_trajectory_programs_and_validation_losses_verified'
    assert json.loads((root / 'layer_allocation_preflight.json').read_text())['status'] == 'instrumented_logits_gradients_tied_use_variances_and_budget_invariants_verified'
    protocol = {'sources_sha256': {n: sha(Path(__file__).with_name(n)) for n in
        ['calibrate_layer_allocation.py', 'layer_allocation.py', 'probability_operators.py', 'probability_tensor.py', 'recomputed_probability.py', 'verify_layer_calibration.py', 'layer_calibration_pipeline.py']},
        'prerequisites_sha256': {n: sha(root / n) for n in ['adaptation_pilot/mean/independent_audit.json',
            'adaptation_pilot/mean/program_package.json', 'adaptation_data/tokens.npz', 'layer_allocation_preflight.json', 'meta_inventory.json']},
        'model_revision': 'ac2586a34f1a1c03329dbdcdefabd8b6478ddaa7',
        'source_arm': 'mean', 'data': 'Reserved train windows16..31, first65tokens, two fixed batchesof8. Disjoint from both16-step fitting arms which usewindows0..15, and fromvalidation/test. No model updates.',
        'batch_size': 8, 'reference_samples': 1024, 'sample_options': [16,32,64,128,256,512,1024,2048,4096],
        'proxy': 'For each independent module-use draw, V=sum(g_w**2*span**2*p*(1-p)) from the declared ACT12/finite-control STE mean surrogate. Average V across two calibrationbatches; representative batchloss-variance proxy=sum(V/T). Separate tiedlookup andheaduse gradients; no combined-gradient cross-term.',
        'allocation': 'Greedy doubling from16 to4096 by variance reduction per extra schedulednodeevaluation. Embedding remains1024 to keep variable lookup cost identical across different inputs. Useactual64-channelpaddinggeometry; budget uppercap equalsuniform1024. Reportunusedbudget andrawgreedyproposal; fallbacktouniform if greedyproxyworse. Not exactbudgetequality orglobaloptimality.',
        'scope': 'First-order STE loss-variance calibration and candidatebudget only. No curvature/bias guarantee, native sampled quality, BF16equivalence or validated allocation improvement. Evaluationrequiresseparateprospectiveprotocol andnativeverification; existinguniformnativeconditionsunchanged.'}
    with (out / 'protocol.json').open('x') as f:
        json.dump(protocol, f, indent=2)
    begin = time.monotonic()
    def progress(stage, **values):
        (out / 'progress.json').write_text(json.dumps({'stage': stage, 'elapsed_seconds': time.monotonic() - begin, **values}) + '\n')
    progress('loading_pinned_backbone')
    model = AutoModelForCausalLM.from_pretrained('data/thrml_spw/lfm3_1p7b/checkpoint',
        trust_remote_code=True, local_files_only=True, dtype=torch.bfloat16, attn_implementation='eager').cpu().eval()
    sys.modules[type(model.model.layers[0].conv).__module__].can_use_lfm3_conv = lambda *args: False
    targets = {name: module for name, module in model.named_modules() if module._parameters.get('weight') is not None}
    kinds = {name: 'embedding' if isinstance(module, torch.nn.Embedding) else 'linear' if isinstance(module, torch.nn.Linear)
        else 'conv' if isinstance(module, torch.nn.Conv1d) else 'elementwise' for name, module in targets.items()}
    lookup = {name for name, kind in kinds.items() if kind == 'embedding'}
    assert lookup == {'model.embed_tokens'}
    state, encoders, aliases = install(model)
    assert len(encoders) == 286 and len(aliases) == 287
    package = json.loads((training / 'program_package.json').read_text())
    programs = {v['name']: v for v in package['programs']}
    assert set(encoders) == set(programs)
    for index, (name, encoder) in enumerate(encoders.items()):
        rec = programs[name]
        path = Path(rec['checkpoint'])
        assert sha(path) == rec['checkpoint_sha256']
        with np.load(path) as saved:
            theta = saved['theta']
            assert array_sha(theta) == rec['state_array_hashes']['theta']
            with torch.no_grad():
                encoder.theta.copy_(torch.from_numpy(theta))
        fp = encoder.fingerprints()
        for key in ['codes_sha256', 'lower_sha256', 'span_sha256']:
            assert fp[key] == rec[key]
        progress('loading_audited_controls', completed=index + 1, last_parameter=name)
    assert sum(e.theta.numel() for e in encoders.values()) == 1728801536
    tokens = np.load(root / 'adaptation_data/tokens.npz')['train_tokens'][16:32, :65]
    assert tokens.shape == (16, 65)
    batch, captures = 0, []
    def capture(owner, canonical, encoder, rows, gradient):
        directory = out / f'batch{batch}'
        directory.mkdir(exist_ok=True)
        path = directory / f'g{len(captures):03d}.npy'
        assert gradient.dtype == torch.float64
        array = np.ascontiguousarray(gradient.detach().numpy())
        with path.open('xb') as f:
            np.save(f, array)
        value, norm = variance_coefficient(encoder, rows, gradient)
        captures.append({'module': owner, 'canonical': canonical, 'kind': kinds[owner],
            'rows': None if rows is None else rows.detach().numpy().tolist(),
            'gradient_path': str(path), 'file_sha256': sha(path), 'array_sha256': array_sha(array),
            'shape': list(array.shape), 'numel': array.size, 'variance_coefficient': value, 'squared_gradient_norm': norm})
        progress('capturing_weight_gradients', batch=batch, completed=len(captures), last_module=owner)
        return gradient
    originals = instrument(encoders, aliases, lookup, capture)
    records = []
    try:
        for batch, start in enumerate(range(0, 16, 8)):
            captures = []
            state['calls'].clear()
            model.zero_grad(set_to_none=True)
            ids = torch.from_numpy(tokens[start:start + 8])
            progress('mean_forward_backward', batch=batch)
            logits = model(input_ids=ids, use_cache=False).logits.float()
            assert Counter(state['calls']) == Counter(aliases.keys()) and torch.isfinite(logits).all()
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1))
            assert torch.isfinite(loss)
            loss.backward()
            assert len(captures) == 287 and {v['module'] for v in captures} == set(aliases)
            assert all(e.theta.grad is not None and torch.isfinite(e.theta.grad).all() for e in encoders.values())
            costs = {}
            for name, canonical in aliases.items():
                shape = programs[canonical]['shape']; kind = kinds[name]
                if kind == 'embedding':
                    fanin, channels = 1, len(np.unique(ids.numpy())) * shape[1]
                elif kind == 'linear':
                    fanin, channels = shape[1], shape[0]
                elif kind == 'conv':
                    fanin, channels = 7, shape[0]
                else:
                    fanin, channels = 1, shape[0]
                costs[name] = fanin * ((channels + 63) // 64 * 64)
            with torch.no_grad():
                nll = F.cross_entropy(logits[:, :-1].double().reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1), reduction='none').reshape(8, 64).numpy()
            path = out / f'batch{batch}_outputs.npz'
            np.savez(path, input_ids=ids.numpy(), logits=logits.detach().numpy(), token_NLL=nll)
            record = {'batch': batch, 'gradient_records': captures, 'costs_including_padding': costs,
                'outputs': str(path), 'outputs_sha256': sha(path), 'loss_FP32_backward': float(loss.detach()),
                'mean_NLL_FP64': float(nll.mean())}
            with (out / f'batch{batch}.json').open('x') as f:
                json.dump(record, f, indent=2)
            records.append(record)
            del logits, loss
        for name, encoder in encoders.items():
            encoder.weights = originals[name]
            assert encoder.fingerprints()['codes_sha256'] == programs[name]['codes_sha256']
            assert array_sha(encoder.theta.detach().numpy()) == programs[name]['state_array_hashes']['theta']
        scores = {name: sum(next(v['variance_coefficient'] for v in rec['gradient_records'] if v['module'] == name) for rec in records) / 2 for name in aliases}
        costs = {name: sum(rec['costs_including_padding'][name] for rec in records) // 2 for name in aliases}
        assert all(records[0]['costs_including_padding'][name] == records[1]['costs_including_padding'][name] for name in aliases if name not in lookup)
        proposal = allocate(scores, costs, 'model.embed_tokens')
        result = {'status': 'full_model_separate_use_gradient_calibration_and_budget_candidate_complete',
            'unique_parameter_coordinates': 1728801536, 'module_uses_per_batch': 287, 'calibration_targets': 1024,
            'module_aliases': aliases, 'kinds': kinds, 'scores': scores, 'mean_costs': costs, 'proposal': proposal,
            'batch_records_sha256': {str(out / f'batch{k}.json'): sha(out / f'batch{k}.json') for k in range(2)},
            'elapsed_seconds': time.monotonic() - begin,
            'process_peak_RSS_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            'scope': protocol['scope']}
        with (out / 'results.json').open('x') as f:
            json.dump(result, f, indent=2)
        progress('complete')
        print(json.dumps({k: result[k] for k in ['status', 'calibration_targets', 'elapsed_seconds', 'process_peak_RSS_bytes']}))
    except Exception as error:
        with (out / 'failure.json').open('x') as f:
            json.dump({'status': 'failed_preserved_no_restart', 'type': type(error).__name__, 'message': str(error)}, f, indent=2)
        raise


if __name__ == '__main__':
    main()
