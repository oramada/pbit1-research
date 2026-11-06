"""One full-model CPU mean-surrogate backward on reserved train tokens.

No optimizer step or language-model adaptation result. Native inference and
full-model ACT12 operator training are separate validation stages.
"""
import hashlib
import json
from pathlib import Path
import resource
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import parametrize
from transformers import AutoModelForCausalLM
from .recomputed_probability import RecomputedProbabilityTensor


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda: f.read(16 * 1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def main():
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    root = Path('results/thrml_spw/lfm3_1p7b')
    out = root / 'full_backward_probe'
    out.mkdir(exist_ok=True)
    check = root / 'recomputed_probability_preflight/independent_audit.json'
    assert json.loads(check.read_text())['status'] == 'all15_first_order_gradient_laws_and_saved_storage_records_verified'
    data_audit = root / 'adaptation_data/independent_audit.json'
    assert json.loads(data_audit.read_text())['status'] == 'all_adaptation_windows_split_identity_and_exact65_context_exclusions_verified'
    protocol = {'sources_sha256': {n: sha(Path(__file__).with_name(n)) for n in
        ['probability_tensor.py', 'recomputed_probability.py', 'full_backward_probe.py']},
        'component_audit_sha256': sha(check), 'data_audit_sha256': sha(data_audit),
        'train_tokens_sha256': sha(root / 'adaptation_data/tokens.npz'),
        'model_revision': 'ac2586a34f1a1c03329dbdcdefabd8b6478ddaa7',
        'input_selection': 'First16tokens from first reserved train window,15teacher-forced targets; no test data.',
        'device': 'CPU', 'reference_compute_dtype': 'BF16', 'control_dtype': 'FP32',
        'control_bits': 12, 'optimizer_steps': 0, 'model_mode': 'eval with gradients enabled',
        'scope': 'Full-checkpoint recomputed mean-weight surrogate backward and finite-gradient/coverage/memory records. No optimizer step, adaptation quality, full-model native sampling, ACT12 operator parity or hardware memory inference.'}
    with (out / 'protocol.json').open('x') as f:
        json.dump(protocol, f, indent=2)
    begin = time.monotonic()
    model = AutoModelForCausalLM.from_pretrained('data/thrml_spw/lfm3_1p7b/checkpoint',
        trust_remote_code=True, local_files_only=True, dtype=torch.bfloat16, attn_implementation='eager').cpu().eval()
    sys.modules[type(model.model.layers[0].conv).__module__].can_use_lfm3_conv = lambda *args: False
    assert all(p.device.type == 'cpu' for p in model.parameters())
    targets = [(name, module, module._parameters['weight']) for name, module in model.named_modules()
               if module._parameters.get('weight') is not None]
    for p in model.parameters():
        p.requires_grad_(False)
    encoders, names, aliases = {}, {}, {}
    for name, module, original in targets:
        key = id(original)
        if key not in encoders:
            encoders[key] = RecomputedProbabilityTensor(original)
            names[key] = name + '.weight'
        with torch.no_grad():
            parametrize.register_parametrization(module, 'weight', encoders[key])
        aliases[name] = names[key]
        (out / 'progress.json').write_text(json.dumps({'stage': 'parametrizing', 'modules': len(aliases), 'last_module': name}) + '\n')
    assert len(encoders) == 286 and len(aliases) == 287
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 1728801536
    assert model.model.embed_tokens.parametrizations.weight[0] is model.lm_head.parametrizations.weight[0]
    for e in encoders.values():
        e.forward_calls = 0
    persistent = {v.untyped_storage().data_ptr() for e in encoders.values() for v in [e.theta, e.lower, e.span]}
    saved = {}
    def pack(tensor):
        storage = tensor.untyped_storage()
        key = (storage.data_ptr(), storage.nbytes())
        saved[key] = {'dtype': str(tensor.dtype), 'bytes': storage.nbytes(),
                      'persistent_control_or_range': storage.data_ptr() in persistent}
        return tensor
    ids = torch.from_numpy(np.load(root / 'adaptation_data/tokens.npz')['train_tokens'][:1, :16])
    (out / 'progress.json').write_text(json.dumps({'stage': 'full_forward_backward'}) + '\n')
    try:
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda value: value):
            logits = model(input_ids=ids, use_cache=False).logits.float()
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1))
            assert torch.isfinite(loss)
            loss.backward()
    except Exception as error:
        (out / 'failure.json').write_text(json.dumps({'status': 'failed_preserved_no_restart',
            'exception': type(error).__name__, 'message': str(error),
            'stage': 'CPU_BF16_full_forward_backward'}, indent=2) + '\n')
        raise
    rows = []
    for key, encoder in encoders.items():
        gradient = encoder.theta.grad
        assert gradient is not None and gradient.dtype == torch.float32 and torch.isfinite(gradient).all()
        assert encoder.forward_calls >= 1
        rows.append({'name': names[key], 'shape': list(gradient.shape), 'numel': gradient.numel(),
                     'gradient_dtype': str(gradient.dtype), 'gradient_nonzero': int(torch.count_nonzero(gradient)),
                     'gradient_squared_sum_FP64': float(gradient.double().square().sum()),
                     'gradient_sha256': hashlib.sha256(gradient.numpy().tobytes()).hexdigest(),
                     'forward_uses': encoder.forward_calls})
    assert sum(v['numel'] for v in rows) == 1728801536
    np.savez_compressed(out / 'arrays.npz', input_ids=ids.numpy(), logits=logits.detach().numpy())
    results = {'status': 'all_unique_control_gradients_full_CPU_BF16_backward_finite',
        'trainable_control_slots': 1728801536, 'unique_encoders': 286, 'weighted_modules': 287,
        'embedding_head_encoder_shared': True, 'gradient_records': rows, 'module_aliases': aliases,
        'diagnostic_train_loss': float(loss.detach()), 'arrays_sha256': sha(out / 'arrays.npz'),
        'saved_storage_records': list(saved.values()),
        'saved_transient_storage_bytes': sum(v['bytes'] for v in saved.values() if not v['persistent_control_or_range']),
        'saved_FP64_storage_bytes': sum(v['bytes'] for v in saved.values() if v['dtype'] == 'torch.float64'),
        'process_peak_RSS_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        'elapsed_seconds': time.monotonic() - begin, 'scope': protocol['scope']}
    with (out / 'results.json').open('x') as f:
        json.dump(results, f, indent=2)
    print(json.dumps({k: v for k, v in results.items() if k not in ['gradient_records', 'module_aliases', 'saved_storage_records']}))


if __name__ == '__main__':
    main()
