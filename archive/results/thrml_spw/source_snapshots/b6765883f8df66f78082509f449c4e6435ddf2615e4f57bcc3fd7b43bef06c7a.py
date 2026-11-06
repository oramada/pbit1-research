"""Five fixed conventional RGB fits; validation-only selection and export."""
import copy
import datetime
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import numpy as np
import torch
from torch.nn import functional as F
from torchvision.datasets import CIFAR10
from results.thrml_spw.cifar_cnn_native.model import CifarCNN16, features

ROOT = Path('results/thrml_spw/cifar_cnn_native')
OUT = ROOT/'training'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(state):
    return hashlib.sha256(b''.join(state[k].detach().cpu().numpy().tobytes() for k in sorted(state))).hexdigest()


def write(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2)+'\n')
    temp.replace(path)


def save(path, **arrays):
    temp = path.with_suffix('.tmp')
    with temp.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    temp.replace(path)


def guard(p):
    for mapping in [p['source_sha256'], p['input_sha256']]:
        for path, expected in mapping.items():
            assert sha(path) == expected, path


def prepare():
    OUT.mkdir(exist_ok=True)
    assert not (OUT/'protocol.json').exists()
    rawdir = Path('data/torchvision/cifar-10-batches-py')
    raw = [rawdir/f'data_batch_{i}' for i in range(1, 6)]+[rawdir/'batches.meta']
    assert all(path.exists() for path in raw)
    p = {'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
         'source_sha256': {str(path): sha(path) for path in ROOT.glob('*.py')},
         'input_sha256': {str(path): sha(path) for path in raw},
         'dataset': 'Cached standard CIFAR10 training corpus; existing pilot consumed test images. No new fresh-test claim.',
         'split_seed': 20261030, 'train_images': 45000, 'validation_images': 5000,
         'model_seeds': [0, 1, 2, 3, 4], 'minibatch_seed_root': 1727520,
         'epochs': 20, 'batch_size': 256, 'threads': 4, 'device': 'cpu',
         'architecture': '16x16RGB; Conv3x8 k3pad1/ReLU/maxpool2; Conv8x16 k3pad1/ReLU/maxpool2; linear256x10.3928signedweights/34biases.',
         'preprocessing': 'Original32x32RGB uint8/NHWC toNCHWFP32 /255, then2x2averagepool. No augmentation, fitted normalization or test statistics.',
         'optimizer': 'Adam lr0.003, cosine annealing20epochs to0.0003, no weight decay.',
         'selection': 'Lowest full validationCE among20fixed epochs for each seed; ties keep earliest. All5seeds retained, no extra epochs/rescue/checkpoint reuse or test-label selection.',
         'future_native': 'Conventional preparation only, not native execution. Intended onefreepbit/normalizedsignedweight, per-output minmax metadata/finite8bitfields±4, deterministicACT8/exactpadding/integeroutputaccumulation. Freeze separate native/control/statistical protocols and validate RGB operators before test scoring. Different smaller architecture than the old single-seed CIFAR pilot, not its reproduction.',
         'activation_ranges': 'Input[0,1]; postReLU/maxpool hidden spans1.25times full selected-model validation maxima, lower0. No test activation fitting.',
         'scope': 'Train five independent compact conventional CNNs and audit split/initializations/minibatchorders/selection/checkpoint exports/full validation outputs/ranges. No test scoring, probabilistic inference, fresh generalization or physical systems claim.',
         'environment': {'torch': torch.__version__, 'numpy': np.__version__, 'python': platform.python_version(), 'deterministic_CPU': True}}
    guard(p)
    write(OUT/'protocol.json', p)


def run():
    p = json.loads((OUT/'protocol.json').read_text())
    guard(p)
    assert not (OUT/'manifest.json').exists()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    data = CIFAR10('data/torchvision', train=True, download=False)
    assert len(data) == 50000
    x, y = features(data.data), torch.tensor(data.targets, dtype=torch.long)
    ids = np.random.default_rng(p['split_seed']).permutation(50000)
    tr, va = ids[:45000], ids[45000:]
    save(OUT/'split.npz', train_indices=tr, validation_indices=va)
    save(OUT/'validation.npz', inputs=x[va].numpy(), labels=y[va].numpy())
    m = {'complete': False, 'protocol_sha256': sha(OUT/'protocol.json'), 'rows': [], 'test_scored': False}
    write(OUT/'manifest.json', m)
    started, initials = time.monotonic(), []
    for seed in p['model_seeds']:
        guard(p)
        torch.manual_seed(seed)
        model = CifarCNN16()
        initial = digest(model.state_dict())
        assert initial not in initials
        initials.append(initial)
        assert sum(v.numel() for k, v in model.state_dict().items() if k.endswith('weight')) == 3928
        generator = torch.Generator().manual_seed(p['minibatch_seed_root']+seed)
        opt = torch.optim.Adam(model.parameters(), lr=.003)
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=20, eta_min=.0003)
        history, best, bestce, bestepoch = [], None, float('inf'), None
        for epoch in range(20):
            model.train()
            order = torch.randperm(45000, generator=generator)
            indices = torch.as_tensor(tr)[order]
            total, steps = 0., 0
            for first in range(0, 45000, 256):
                batch = indices[first:first+256]
                opt.zero_grad(set_to_none=True)
                loss = F.cross_entropy(model(x[batch]), y[batch])
                loss.backward()
                opt.step()
                total += float(loss.detach())*len(batch)
                steps += 1
            model.eval()
            ce, correct = 0., 0
            with torch.no_grad():
                for first in range(0, 5000, 256):
                    batch = va[first:first+256]
                    z = model(x[batch])
                    ce += float(F.cross_entropy(z, y[batch], reduction='sum'))
                    correct += int((z.argmax(1) == y[batch]).sum())
            ce /= 5000
            row = {'seed': seed, 'epoch': epoch+1, 'train_CE': total/45000, 'validation_CE': ce,
                   'validation_accuracy': correct/5000, 'optimizer_steps': steps,
                   'permutation_sha256': hashlib.sha256(order.numpy().tobytes()).hexdigest(),
                   'epoch_model_sha256': digest(model.state_dict())}
            history.append(row)
            if ce < bestce:
                bestce, best, bestepoch = ce, copy.deepcopy(model.state_dict()), epoch+1
            schedule.step()
            write(OUT/f'seed{seed}_history.json', history)
            print(json.dumps({**row, 'elapsed_seconds': time.monotonic()-started}), flush=True)
        model.load_state_dict(best)
        model.eval()
        parts, maxima = [], np.zeros(2)
        with torch.no_grad():
            for first in range(0, 5000, 256):
                inputs = x[va[first:first+256]]
                activations = model.activations(inputs)
                maxima = np.maximum(maxima, [float(a.max()) for a in activations])
                parts.append(model(inputs).numpy())
        assert np.all(maxima > 0)
        arrays = {k.replace('.', '_'): v.numpy() for k, v in best.items()}
        arrays.update(reference_validation_logits=np.concatenate(parts), activation_lower=np.zeros(3),
                      activation_span=np.array([1., 1.25*maxima[0], 1.25*maxima[1]]))
        path, checkpoint = OUT/f'seed{seed}.npz', OUT/f'seed{seed}.pt'
        save(path, **arrays)
        temp = checkpoint.with_suffix('.tmp')
        torch.save(best, temp)
        temp.replace(checkpoint)
        m['rows'].append({'seed': seed, 'selected_epoch': bestepoch, 'selected_validation_CE': bestce,
                          'initial_state_sha256': initial, 'selected_state_sha256': digest(best),
                          'path': str(path), 'sha256': sha(path), 'checkpoint_path': str(checkpoint),
                          'checkpoint_sha256': sha(checkpoint), 'history_sha256': sha(OUT/f'seed{seed}_history.json')})
        write(OUT/'manifest.json', m)
    guard(p)
    assert len(m['rows']) == 5 and len({r['selected_state_sha256'] for r in m['rows']}) == 5
    m.update(complete=True, elapsed_seconds=time.monotonic()-started)
    write(OUT/'manifest.json', m)
    print('All5compactRGB conventional fits exported; independent audit pending', flush=True)


if __name__ == '__main__':
    {'prepare': prepare, 'run': run}[sys.argv[1]]()
