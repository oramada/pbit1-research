"""Full split/export/selection/order/validation checks, no full retraining claim."""
import datetime
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torchvision.datasets import CIFAR10
from results.thrml_spw.cifar_cnn_native.model import CifarCNN16
from results.thrml_spw.cifar_cnn_native.train import OUT, sha, digest, guard, write


def main():
    assert not (OUT/'completion.json').exists()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    p = json.loads((OUT/'protocol.json').read_text())
    guard(p)
    m = json.loads((OUT/'manifest.json').read_text())
    assert m['complete'] and not m['test_scored'] and m['protocol_sha256'] == sha(OUT/'protocol.json')
    assert [r['seed'] for r in m['rows']] == [0, 1, 2, 3, 4]
    data = CIFAR10('data/torchvision', train=True, download=False)
    ids = np.random.default_rng(p['split_seed']).permutation(50000)
    split = np.load(OUT/'split.npz')
    np.testing.assert_array_equal(split['train_indices'], ids[:45000])
    np.testing.assert_array_equal(split['validation_indices'], ids[45000:])
    assert np.intersect1d(split['train_indices'], split['validation_indices']).size == 0
    raw = data.data[ids[45000:]].transpose(0, 3, 1, 2).astype(np.float32)/np.float32(255)
    x = np.ascontiguousarray((raw[:, :, 0::2, 0::2]+raw[:, :, 0::2, 1::2]+raw[:, :, 1::2, 0::2]+raw[:, :, 1::2, 1::2])/np.float32(4))
    validation = np.load(OUT/'validation.npz')
    np.testing.assert_array_equal(x, validation['inputs'])
    np.testing.assert_array_equal(np.array(data.targets)[ids[45000:]], validation['labels'])
    rows = []
    for row in m['rows']:
        seed = row['seed']
        torch.manual_seed(seed)
        model = CifarCNN16()
        assert digest(model.state_dict()) == row['initial_state_sha256']
        assert sha(row['path']) == row['sha256'] and sha(row['checkpoint_path']) == row['checkpoint_sha256']
        historypath = OUT/f'seed{seed}_history.json'
        assert sha(historypath) == row['history_sha256']
        history = json.loads(historypath.read_text())
        assert len(history) == 20
        chosen = min(history, key=lambda r: r['validation_CE'])
        assert chosen['epoch'] == row['selected_epoch'] and chosen['validation_CE'] == row['selected_validation_CE']
        assert chosen['epoch_model_sha256'] == row['selected_state_sha256']
        generator = torch.Generator().manual_seed(p['minibatch_seed_root']+seed)
        for epoch, item in enumerate(history, 1):
            order = torch.randperm(45000, generator=generator)
            assert hashlib.sha256(order.numpy().tobytes()).hexdigest() == item['permutation_sha256']
            assert item['optimizer_steps'] == 176 and item['epoch'] == epoch
        state = torch.load(row['checkpoint_path'], map_location='cpu', weights_only=True)
        assert digest(state) == row['selected_state_sha256']
        arrays = np.load(row['path'])
        for name, value in state.items():
            np.testing.assert_array_equal(value.numpy(), arrays[name.replace('.', '_')])
        model.load_state_dict(state)
        model.eval()
        parts, maxima = [], np.zeros(2)
        with torch.no_grad():
            for first in range(0, 5000, 256):
                xx = torch.from_numpy(x[first:first+256])
                a = F.max_pool2d(F.relu(F.conv2d(xx, torch.from_numpy(arrays['conv1_weight']), torch.from_numpy(arrays['conv1_bias']), padding=1)), 2)
                b = F.max_pool2d(F.relu(F.conv2d(a, torch.from_numpy(arrays['conv2_weight']), torch.from_numpy(arrays['conv2_bias']), padding=1)), 2)
                z = F.linear(b.flatten(1), torch.from_numpy(arrays['head_weight']), torch.from_numpy(arrays['head_bias']))
                np.testing.assert_array_equal(z.numpy(), model(xx).numpy())
                parts.append(z.numpy())
                maxima = np.maximum(maxima, [float(a.max()), float(b.max())])
        logits = np.concatenate(parts)
        np.testing.assert_array_equal(logits, arrays['reference_validation_logits'])
        np.testing.assert_array_equal(arrays['activation_lower'], [0, 0, 0])
        np.testing.assert_array_equal(arrays['activation_span'], [1., 1.25*maxima[0], 1.25*maxima[1]])
        ce = sum(float(F.cross_entropy(torch.from_numpy(logits[i:i+256]), torch.from_numpy(validation['labels'][i:i+256]), reduction='sum')) for i in range(0, 5000, 256))/5000
        assert ce == row['selected_validation_CE']
        rows.append({'seed': seed, 'selected_epoch': row['selected_epoch'], 'validation_CE': ce,
                     'validation_accuracy': float(np.mean(logits.argmax(1) == validation['labels'])),
                     'all50000validation_logits_exact': True, 'all20minibatchorders_exact': True})
    assert len({r['initial_state_sha256'] for r in m['rows']}) == 5
    assert len({r['selected_state_sha256'] for r in m['rows']}) == 5
    guard(p)
    write(OUT/'evidence_audit.json', {'passed': True, 'recorded_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                    'rows': rows, 'validation_logit_values_checked': 250000, 'permutations_checked': 100,
                                    'scope': 'All5checkpoint initializations/exports/splits/selection/100minibatchorders/fullvalidation outputs/ranges. Independent preprocessing and explicit functional graph, same Torch primitives. Not full independent retraining, native inference, test scoring or fresh generalization.'})
    paths = [OUT/n for n in ['protocol.json', 'manifest.json', 'split.npz', 'validation.npz', 'evidence_audit.json']]
    for row in m['rows']:
        paths += [OUT/f'seed{row["seed"]}_history.json', Path(row['path']), Path(row['checkpoint_path'])]
    write(OUT/'completion.json', {'recorded_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                 'status': 'all5conventionalRGBcheckpoints_and_validation_exports_audited_native_pending_goal_active',
                                 'artifact_sha256': {str(path): sha(path) for path in paths}})
    print('All5RGBcheckpoints/250000validation logits/100minibatchorders audited', flush=True)


if __name__ == '__main__':
    main()
