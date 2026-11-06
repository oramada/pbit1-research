"""Independent exports, selection, orders and selected surrogate validation."""
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path('results/thrml_spw/cifar_cnn_native')
OUT = ROOT/'probability_finetune_evidence'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reference(x, arrays, generator):
    h = x
    spans = torch.tensor(arrays['activation_span'], dtype=torch.float32)
    for li, name in enumerate(['conv1', 'conv2', 'head']):
        bound = (h/spans[li]).clamp(0, 1)
        q = torch.round(255*bound)/255
        h = spans[li]*(bound+(q-bound))
        latent = torch.tensor(arrays[name+'_latent_fields'], dtype=torch.float32)
        field = -4+torch.round((latent+4)*255/8)*8/255
        effective = latent+(field-latent)
        probability = torch.sigmoid(2*effective)
        c = torch.tensor(arrays[name+'_center'], dtype=torch.float32)[:, None]
        r = torch.tensor(arrays[name+'_radius'], dtype=torch.float32)[:, None]
        mu = c-r+2*r*probability
        draw = torch.binomial(torch.full_like(probability, 256), probability, generator=generator)
        sampled = c-r+2*r*draw/256
        weight = (mu+(sampled-mu)).reshape(arrays[name+'_weight'].shape)
        bias = torch.tensor(arrays[name+'_bias'])
        h = F.max_pool2d(F.relu(F.conv2d(h, weight, bias, padding=1)), 2) if li < 2 else F.linear(h.flatten(1), weight, bias)
    return h.numpy()


def main():
    assert not (OUT/'evidence_audit.json').exists()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    p = json.loads((OUT/'protocol.json').read_text())
    for mapping in [p['source_sha256'], p['input_sha256']]:
        for path, expected in mapping.items():
            assert sha(path) == expected, path
    manifest = json.loads((OUT/'manifest.json').read_text())
    assert manifest['complete'] and not manifest['test_scored'] and manifest['native_updates'] == 0
    assert len(manifest['rows']) == 10
    assert {(r['model'], r['method']) for r in manifest['rows']} == {(m, name) for m in range(5) for name in p['methods']}
    training = json.loads((ROOT/'training/manifest.json').read_text())
    data = np.load(ROOT/'training/validation.npz')
    x, y = torch.from_numpy(data['inputs']), data['labels']
    max_probability_difference = 0.
    rows = []
    for item in manifest['rows']:
        mi, method = item['model'], item['method']
        assert sha(item['path']) == item['sha256']
        history_path = OUT/f'model{mi}_{method}_history.json'
        assert sha(history_path) == item['history_sha256']
        history = json.loads(history_path.read_text())
        assert len(history) == 9 and [r['epoch'] for r in history] == list(range(9))
        chosen = int(np.argmin([r['validation_CE'] for r in history]))
        assert item['selected_epoch'] == chosen and item['validation_CE'] == history[chosen]['validation_CE']
        generator = torch.Generator().manual_seed(p['order_seed_root']+mi)
        for row in history[1:]:
            order = torch.randperm(45000, generator=generator)
            assert hashlib.sha256(order.numpy().tobytes()).hexdigest() == row['order_sha256']
            assert row['steps'] == 176
        original = np.load(training['rows'][mi]['path'])
        arrays = np.load(item['path'])
        np.testing.assert_array_equal(arrays['activation_span'], original['activation_span'].astype(np.float32).astype(float))
        for name in ['conv1', 'conv2', 'head']:
            source = original[name+'_weight'].astype(float).reshape(original[name+'_weight'].shape[0], -1)
            c, r = (source.max(1)+source.min(1))/2, (source.max(1)-source.min(1))/2
            np.testing.assert_array_equal(c, arrays[name+'_center'])
            np.testing.assert_array_equal(r, arrays[name+'_radius'])
            np.testing.assert_array_equal(original[name+'_bias'], arrays[name+'_bias'])
            latent = arrays[name+'_latent_fields']
            assert np.isfinite(latent).all() and np.max(abs(latent)) <= 4
            code = np.rint((latent+4)*255/8).astype(np.uint8)
            np.testing.assert_array_equal(code, arrays[name+'_commands'])
            field = (-4+code.astype(float)*8/255).astype(np.float32)
            np.testing.assert_array_equal(field, arrays[name+'_fields'])
            nominal_p = 1/(1+np.exp(-2*field.astype(float)))
            expected = (c[:, None]-r[:, None]+2*r[:, None]*nominal_p).reshape(source.shape)
            np.testing.assert_allclose(expected, arrays[name+'_weight'].reshape(source.shape), rtol=2e-14, atol=2e-15)
            h = torch.tensor(latent, dtype=torch.float32)
            q = -4+torch.round((h+4)*255/8)*8/255
            train_p = torch.sigmoid(2*(h+(q-h))).numpy()
            max_probability_difference = max(max_probability_difference, float(np.max(abs(train_p-nominal_p))))
        replicas = []
        for seed in range(4):
            rng = torch.Generator().manual_seed(p['validation_sample_seed_root']+100*mi+seed)
            replicas.append(np.concatenate([reference(x[first:first+256], arrays, rng) for first in range(0, 5000, 256)]))
        z = np.stack(replicas)
        np.testing.assert_array_equal(z, arrays['selected_validation_logits'])
        z = z.astype(float)
        shift = z-z.max(2, keepdims=True)
        ce = float(np.mean(np.log(np.exp(shift).sum(2))-np.take_along_axis(shift, y[None, :, None], axis=2)[:, :, 0]))
        acc = float(np.mean(z.argmax(2)==y[None, :]))
        np.testing.assert_allclose([ce, acc], [item['validation_CE'], history[chosen]['validation_accuracy']], rtol=2e-13, atol=2e-14)
        rows.append({'model': mi, 'method': method, 'selected_epoch': chosen, 'validation_CE': ce, 'validation_accuracy': acc})
    result = {'passed': True, 'recorded_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'source_sha256': sha(__file__), 'manifest_sha256': sha(OUT/'manifest.json'), 'rows': rows,
              'selected_validation_logit_values_exact': 2000000, 'epoch_orders_verified': 80,
              'optimizer_steps_recorded': 14080, 'max_train_vs_nominal_probability_difference': max_probability_difference,
              'new_native_updates': 0, 'test_scored': False,
              'scope': 'Independent functional replay of all selected validation outputs and audit of frozen exports/selection/history/orders. No independent full retraining or raw THRML execution. F32 training grid/probability arithmetic differs slightly from exported nominal F64-law controls; measured difference retained. Test/native confirmation remains required.'}
    (OUT/'evidence_audit.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
