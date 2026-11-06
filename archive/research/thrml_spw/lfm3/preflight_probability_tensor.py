"""All-coordinate CPU parametrization and tied-control integration; no training."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
from torch.nn.utils import parametrize
from transformers import AutoModelForCausalLM
from .probability_tensor import ProbabilityTensor
from .probability_training import ProbabilityLinear


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda: f.read(16 * 1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def main():
    torch.set_num_threads(2)
    root = Path('results/thrml_spw/lfm3_1p7b')
    out = root / 'probability_tensor_integration'
    out.mkdir(exist_ok=True)
    checkpoint = Path('data/thrml_spw/lfm3_1p7b/checkpoint')
    protocol = {'sources_sha256': {n: sha(Path(__file__).with_name(n)) for n in ['probability_tensor.py', 'preflight_probability_tensor.py', 'probability_training.py']},
                'inventory_sha256': sha(root / 'meta_inventory.json'),
                'checkpoint_inventory_sha256': sha(root / 'preparation_inventory.json'),
                'baseline_input_sha256': sha(root / 'baseline_smoke.npz'),
                'representation': 'One FP32 trainable latent/control per unique original weight, initialized at decoded12-bit fields; fixed FP32 lower/span per block128 or seven-weight kernel. Shared embedding/head uses one encoder and theta storage. Original BF16 tensors retained frozen as references.',
                'gradient': 'Declared identity STE through finite command encoding, explicit project() required after optimizer updates.',
                'scope': 'CPU full-checkpoint mapping/command roundtrip and finite mean-weight surrogate forward, plus all four operator anchors and representative tied-gradient addition. No full-model backward, training, ACT12 full-model surrogate operator integration, native sampling, task-quality or memory/hardware improvement claim.'}
    with (out / 'protocol.json').open('x') as f:
        json.dump(protocol, f, indent=2)
    checkpoint_hashes = json.loads((root / 'preparation_inventory.json').read_text())['checkpoint_files']
    for name, digest in checkpoint_hashes.items():
        assert sha(checkpoint / name) == digest
    anchors = {}
    anchor_rows = []
    for kind, index, reference in [('linear', 0, 24), ('conv', 4, 26), ('embedding', 8, 28), ('elementwise', 12, 30)]:
        path = root / 'group128_preflight' / f'input_{index}.npz'
        a = np.load(path)
        w = a['w'].T if kind == 'linear' else a['w'].T[:, None, :] if kind == 'conv' else a['w'].reshape(2, 2048) if kind == 'embedding' else a['w'].reshape(-1)
        encoder = ProbabilityTensor(torch.from_numpy(w.copy()).to(torch.bfloat16))
        with torch.no_grad():
            mean, variance = encoder.moments()
            _, _, x = ProbabilityLinear.quantize_inputs(torch.from_numpy(a['x']))
            wm = mean.T if kind == 'linear' else mean[:, 0, :].T if kind == 'conv' else mean.reshape(1, -1)
            y = x @ wm if kind == 'linear' else (x * wm[None]).sum(1)
        target = root / 'precision_preflight' / f'output_{reference}.npy'
        np.testing.assert_allclose(y.numpy(), np.load(target), rtol=2e-12, atol=2e-12)
        anchors[kind + '_codes'] = encoder.commands()[0].numpy()
        anchors[kind + '_lower'] = encoder.lower.numpy()
        anchors[kind + '_span'] = encoder.span.numpy()
        anchors[kind + '_mean'] = mean.numpy()
        anchors[kind + '_variance'] = variance.numpy()
        anchors[kind + '_output'] = y.numpy()
        anchor_rows.append({'kind': kind, 'input_path': str(path), 'input_sha256': sha(path),
                            'reference_path': str(target), 'reference_sha256': sha(target),
                            'shape': list(w.shape)})
    # Actual parametrized Embedding/Linear calls, with one shared encoder.
    toy = torch.nn.Module()
    toy.embedding = torch.nn.Embedding(2, 2048, dtype=torch.bfloat16)
    toy.head = torch.nn.Linear(2048, 2, bias=False, dtype=torch.bfloat16)
    with torch.no_grad():
        toy.embedding.weight.copy_(torch.from_numpy(np.load(root / 'group128_preflight/input_8.npz')['w'].reshape(2, 2048)))
    toy.head.weight = toy.embedding.weight
    toy.embedding.weight.requires_grad_(False)
    shared = ProbabilityTensor(toy.embedding.weight)
    with torch.no_grad():
        parametrize.register_parametrization(toy.embedding, 'weight', shared)
        parametrize.register_parametrization(toy.head, 'weight', shared)
    assert toy.embedding.parametrizations.weight[0] is toy.head.parametrizations.weight[0]
    assert sum(p.numel() for p in toy.parameters() if p.requires_grad) == 4096
    le = toy.embedding(torch.tensor([0, 1, 0])).float().sum()
    lh = toy.head(torch.linspace(-1, 1, 2048).to(torch.bfloat16)[None]).float().sum()
    ge = torch.autograd.grad(le, shared.theta, retain_graph=True)[0]
    gh = torch.autograd.grad(lh, shared.theta, retain_graph=True)[0]
    combined = torch.autograd.grad(le + lh, shared.theta)[0]
    torch.testing.assert_close(combined, ge + gh, rtol=1e-6, atol=1e-6)
    anchors.update(tied_embedding_gradient=ge.numpy(), tied_head_gradient=gh.numpy(), tied_combined_gradient=combined.numpy())
    np.savez_compressed(out / 'anchors.npz', **anchors)
    (out / 'anchor_results.json').write_text(json.dumps({'anchors': anchor_rows, 'shared_trainable_controls': 4096,
        'both_paths_contribute': bool(torch.count_nonzero(ge) and torch.count_nonzero(gh)),
        'arrays_sha256': sha(out / 'anchors.npz')}, indent=2) + '\n')
    del toy, shared, anchors, ge, gh, combined, le, lh
    model = AutoModelForCausalLM.from_pretrained(checkpoint, trust_remote_code=True, local_files_only=True,
        dtype=torch.bfloat16, attn_implementation='eager').cpu().eval()
    sys.modules[type(model.model.layers[0].conv).__module__].can_use_lfm3_conv = lambda *args: False
    assert all(p.device.type == 'cpu' for p in model.parameters())
    assert model.model.embed_tokens.weight is model.lm_head.weight
    ids = torch.from_numpy(np.load(root / 'baseline_smoke.npz')['input_ids'])
    with torch.inference_mode():
        baseline = model(input_ids=ids, use_cache=False).logits.float().numpy()
    targets = [(name, module, module._parameters['weight']) for name, module in model.named_modules() if module._parameters.get('weight') is not None]
    unique_original = {id(p): p for _, _, p in targets}
    assert sum(p.numel() for p in unique_original.values()) == 1728801536
    for p in model.parameters():
        p.requires_grad_(False)
    encoders, canonical, records, aliases = {}, {}, [], {}
    for name, module, original in targets:
        key = id(original)
        if key not in encoders:
            encoder = ProbabilityTensor(original)
            encoders[key] = encoder
            canonical[key] = name + '.weight'
            fingerprints = encoder.fingerprints()
            assert fingerprints['codes_sha256'] == fingerprints['initial_codes_sha256']
            records.append({'canonical_parameter': canonical[key], 'shape': list(original.shape),
                'numel': original.numel(), 'group_size': encoder.group_size,
                'groups': encoder.lower.numel(), 'theta_dtype': str(encoder.theta.dtype),
                'metadata_dtype': str(encoder.lower.dtype), **fingerprints})
        aliases[name] = canonical[key]
        with torch.no_grad():
            parametrize.register_parametrization(module, 'weight', encoders[key])
        (out / 'progress.json').write_text(json.dumps({'stage': 'parametrizing', 'modules': len(aliases), 'last_module': name}) + '\n')
    assert len(encoders) == 286 and len(aliases) == 287
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 1728801536
    assert sum(p.numel() for p in model.parameters() if not p.requires_grad) == 1728801536
    assert sum(e.lower.numel() for e in encoders.values()) == 13531430
    assert model.model.embed_tokens.parametrizations.weight[0] is model.lm_head.parametrizations.weight[0]
    for encoder in encoders.values():
        encoder.forward_calls = 0
    (out / 'progress.json').write_text(json.dumps({'stage': 'full_CPU_mean_weight_forward'}) + '\n')
    with torch.inference_mode():
        logits = model(input_ids=ids, use_cache=False).logits.float().numpy()
    assert np.isfinite(logits).all()
    for record in records:
        key = next(k for k, name in canonical.items() if name == record['canonical_parameter'])
        record['surrogate_forward_uses'] = encoders[key].forward_calls
        assert record['surrogate_forward_uses'] >= 1
    np.savez_compressed(out / 'forward_arrays.npz', input_ids=ids.numpy(), CPU_BF16=baseline, CPU_mean_weight_surrogate=logits)
    results = {'status': 'all_unique_controls_and_full_CPU_mean_weight_forward_verified',
        'original_unique_learned_weights': 1728801536, 'trainable_probability_controls': 1728801536,
        'frozen_original_reference_weights_retained': 1728801536, 'range_groups': 13531430,
        'weighted_modules': 287, 'unique_encoders': 286, 'embedding_head_encoder_shared': True,
        'encoders': records, 'module_aliases': aliases, 'forward_arrays_sha256': sha(out / 'forward_arrays.npz'),
        'CPU_mean_weight_logit_MSE_vs_CPU_BF16': float(np.mean((logits.astype(np.float64) - baseline) ** 2)),
        'CPU_mean_weight_argmax_agreement': float(np.mean(logits.argmax(-1) == baseline.argmax(-1))),
        'scope': protocol['scope']}
    with (out / 'results.json').open('x') as f:
        json.dump(results, f, indent=2)
    print(json.dumps({k: v for k, v in results.items() if k not in ['encoders', 'module_aliases']}))


if __name__ == '__main__':
    main()
