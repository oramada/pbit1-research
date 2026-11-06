"""Audit saved full-backward coverage records and diagnostic loss precision."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda: f.read(16 * 1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def main():
    torch.set_num_threads(2)
    root = Path('results/thrml_spw/lfm3_1p7b')
    out = root / 'full_backward_probe'
    p = json.loads((out / 'protocol.json').read_text())
    r = json.loads((out / 'results.json').read_text())
    for name, digest in p['sources_sha256'].items():
        assert sha(Path(__file__).with_name(name)) == digest
    assert p['optimizer_steps'] == 0 and p['device'] == 'CPU'
    assert sha(root / 'recomputed_probability_preflight/independent_audit.json') == p['component_audit_sha256']
    assert sha(root / 'adaptation_data/independent_audit.json') == p['data_audit_sha256']
    assert sha(root / 'adaptation_data/tokens.npz') == p['train_tokens_sha256']
    assert r['status'] == 'all_unique_control_gradients_full_CPU_BF16_backward_finite'
    meta = json.loads((root / 'meta_inventory.json').read_text())
    expected = {v['name']: v for v in meta['parameters']}
    records = {v['name']: v for v in r['gradient_records']}
    assert len(records) == len(expected) == r['unique_encoders'] == 286
    assert set(records) == set(expected)
    nonzero = 0
    squared_norm = 0.
    for name, rec in records.items():
        assert rec['shape'] == expected[name]['shape'] and rec['numel'] == expected[name]['numel']
        assert rec['gradient_dtype'] == 'torch.float32' and rec['forward_uses'] >= 1
        assert 0 <= rec['gradient_nonzero'] <= rec['numel']
        value = rec['gradient_squared_sum_FP64']
        assert math.isfinite(value) and value >= 0
        assert (rec['gradient_nonzero'] == 0) == (value == 0)
        assert len(rec['gradient_sha256']) == 64 and all(v in '0123456789abcdef' for v in rec['gradient_sha256'])
        nonzero += rec['gradient_nonzero']; squared_norm += value
    assert sum(v['numel'] for v in records.values()) == r['trainable_control_slots'] == 1728801536
    aliases = r['module_aliases']
    assert set(aliases) == {name.removesuffix('.weight') for name in expected} | {'lm_head'}
    assert len(aliases) == r['weighted_modules'] == 287 and r['embedding_head_encoder_shared']
    for name, canonical in aliases.items():
        assert canonical == ('model.embed_tokens.weight' if name == 'lm_head' else name + '.weight')
    assert records['model.embed_tokens.weight']['forward_uses'] >= 2
    assert r['saved_transient_storage_bytes'] == sum(v['bytes'] for v in r['saved_storage_records'] if not v['persistent_control_or_range'])
    assert r['saved_FP64_storage_bytes'] == sum(v['bytes'] for v in r['saved_storage_records'] if v['dtype'] == 'torch.float64') == 0
    assert r['process_peak_RSS_bytes'] > 0 and r['elapsed_seconds'] > 0
    assert sha(out / 'arrays.npz') == r['arrays_sha256']
    arrays = np.load(out / 'arrays.npz')
    source = np.load(root / 'adaptation_data/tokens.npz')['train_tokens'][:1, :16]
    np.testing.assert_array_equal(arrays['input_ids'], source)
    y = arrays['logits']
    assert y.shape == (1, 16, 128000) and y.dtype == np.float32 and np.isfinite(y).all()
    logits = torch.from_numpy(y[:, :-1].reshape(-1, 128000))
    target = torch.from_numpy(source[:, 1:].reshape(-1))
    f32 = F.cross_entropy(logits, target).item()
    # Same F32 operator, with a predetermined four-ULP roundoff allowance.
    assert abs(f32 - r['diagnostic_train_loss']) <= 4 * float(np.spacing(np.float32(f32)))
    f64 = F.cross_entropy(logits.double(), target).item()
    checkpoint_inventory = json.loads((root / 'preparation_inventory.json').read_text())
    actual_checkpoint_sha = sha(Path('data/thrml_spw/lfm3_1p7b/checkpoint/model.safetensors'))
    assert actual_checkpoint_sha == checkpoint_inventory['checkpoint_files']['model.safetensors']
    audit = {'status': 'all_control_gradient_coverage_records_saved_loss_and_memory_records_verified',
             'source_sha256': sha(Path(__file__)), 'results_sha256': sha(out / 'results.json'),
             'trainable_control_slots': 1728801536, 'nonzero_gradient_coordinates': nonzero,
             'reported_global_gradient_L2': math.sqrt(squared_norm),
             'diagnostic_loss_reproduced_FP32': f32, 'diagnostic_loss_recomputed_FP64': f64,
             'saved_FP64_storage_bytes': 0, 'process_peak_RSS_bytes': r['process_peak_RSS_bytes'],
             'current_checkpoint_sha256': actual_checkpoint_sha,
             'scope': 'Coverage, per-parameter finite-gradient producer records/digests, source/data hashes, saved-loss arithmetic and memory telemetry checked. Does not independently rerun the full backward or reconstruct unpersisted6.9GBgradient arrays; first-order component algebra audited separately. One15-target train diagnostic, no optimizer step, native sampler gradient, full-model ACT12 parity, adaptation quality or hardware memory advantage.'}
    with (out / 'independent_audit.json').open('x') as f:
        json.dump(audit, f, indent=2)
    print(json.dumps(audit))


if __name__ == '__main__':
    main()
