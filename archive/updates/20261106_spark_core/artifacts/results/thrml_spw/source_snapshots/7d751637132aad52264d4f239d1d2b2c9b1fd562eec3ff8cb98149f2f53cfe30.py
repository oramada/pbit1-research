"""Validation-selected CPU probability fine-tuning, with no test/native claims."""
import copy
import datetime
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torchvision.datasets import FashionMNIST

from results.thrml_spw.fashion_cnn_native.model import features
from results.thrml_spw.fashion_cnn_native.probability_finetune.model import ProbabilityCNN

ROOT = Path('results/thrml_spw/fashion_cnn_native')
OUT = ROOT / 'probability_finetune_evidence'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, obj):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2)+'\n')
    tmp.replace(path)


def save(path, **arrays):
    tmp = path.with_suffix('.tmp')
    with tmp.open('wb') as f:
        np.savez_compressed(f, **arrays)
    tmp.replace(path)


def guard(protocol):
    for mapping in [protocol['source_sha256'], protocol['input_sha256']]:
        for path, expected in mapping.items():
            assert sha(path) == expected, path


def freeze():
    OUT.mkdir(exist_ok=True)
    assert not (OUT/'protocol.json').exists()
    sources = list(Path(__file__).parent.glob('*.py')) + [ROOT/'model.py']
    original = json.loads((ROOT/'training/manifest.json').read_text())
    inputs = [ROOT/'training/manifest.json', ROOT/'training/split.npz', ROOT/'training/validation.npz',
              ROOT/'cohort_evidence/completion.json']
    inputs += [Path(row['path']) for row in original['rows']]
    # No test data are read by the training program; parent closure is provenance.
    inputs += list(Path('data/torchvision/FashionMNIST/raw').glob('train-*-ubyte'))
    write(OUT/'protocol.json', {
        'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'source_sha256': {str(p): sha(p) for p in sources},
        'input_sha256': {str(p): sha(p) for p in inputs},
        'models': 5, 'methods': ['expected_weight', 'binomial_mean_STE_T256'],
        'epochs': 8, 'batch': 256, 'validation_batch': 256, 'validation_replicates': 4,
        'validation_samples': 256, 'learning_rate': .01, 'final_learning_rate': .001,
        'order_seed_root': 1607520, 'training_sample_seed_root': 1617520, 'validation_sample_seed_root': 1627520,
        'selection': 'Each of10model/method fits retains lowest mean per-example CE across four fixed-seed binomial-mean validation replicas atT256, epoch0through8. Include allfits even if epoch0wins; no test selection.',
        'fixed_parameters': 'Original per-output minmax centers/radii and deterministic biases. Original activation spans cast toF32 for training/export and frozen. Only field latents optimized; project[-4,4] afterAdam, nearest-even8bitfield command forward with identitySTE.',
        'training': 'TorchCPU4threads deterministic, Adam/cosine8epochs, no weight decay. ACT8roundingSTE with ordinary bounded clamp derivative. Binomial mean weight forward shares one realization across minibatch, independent betweenlayers/batches; biased expected-weight derivative, not REINFORCE or officialTHRML execution. Expected method uses stationary mean weights. Both get identical model-specific minibatch orders; sample streams separate.',
        'export': 'Original fixed F64centers/radii, uint8commands/F32fields and latentfields; never refit minmax to expected weights. Expected F64weights provided for deterministic controls, not a new representation. Explicit native_encoding adapter required.',
        'scope': 'Exploratory post-baseline probability fine-tuning afterCNN quality-target miss, not a retroactive rescue or fresh-data confirmation. No officialnative sampling/test task scoring here; surrogate training/validation only. Fullnative causal/audit/statistical confirmation remains required and separately source-frozen before test execution.',
        'environment': {'torch': torch.__version__, 'numpy': np.__version__, 'device': 'cpu'}
    })


def validate(model, x, y, protocol, model_id):
    model.eval()
    logits = []
    with torch.no_grad():
        for replicate in range(protocol['validation_replicates']):
            generator = torch.Generator().manual_seed(protocol['validation_sample_seed_root']+100*model_id+replicate)
            parts = [model(x[first:first+protocol['validation_batch']], protocol['validation_samples'], generator).numpy()
                     for first in range(0, len(x), protocol['validation_batch'])]
            logits.append(np.concatenate(parts))
    z = np.stack(logits).astype(float)
    shifted = z-z.max(2, keepdims=True)
    ce = np.mean(np.log(np.exp(shifted).sum(2))-np.take_along_axis(shifted, y.numpy()[None, :, None], axis=2)[:, :, 0])
    accuracy = np.mean(z.argmax(2)==y.numpy()[None, :])
    return float(ce), float(accuracy), z.astype(np.float32)


def run():
    p = json.loads((OUT/'protocol.json').read_text())
    guard(p)
    assert not (OUT/'manifest.json').exists(), 'No implicit duplicate fine-tuning'
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    assert torch.__version__ == p['environment']['torch'] and np.__version__ == p['environment']['numpy']
    dataset = FashionMNIST('data/torchvision', train=True, download=False)
    x, y = features(dataset.data), dataset.targets
    split = np.load(ROOT/'training/split.npz')
    tr, va = split['train_indices'], split['validation_indices']
    assert len(tr) == 55000 and len(va) == 5000 and not np.intersect1d(tr, va).size
    np.testing.assert_array_equal(x[va].numpy(), np.load(ROOT/'training/validation.npz')['inputs'])
    original = json.loads((ROOT/'training/manifest.json').read_text())
    state = {'complete': False, 'protocol_sha256': sha(OUT/'protocol.json'), 'rows': [], 'test_scored': False, 'native_updates': 0}
    write(OUT/'manifest.json', state)
    started = time.monotonic()
    for mi, row in enumerate(original['rows']):
        for method in p['methods']:
            guard(p)
            initial = np.load(row['path'])
            model = ProbabilityCNN(initial)
            order_rng = torch.Generator().manual_seed(p['order_seed_root']+mi)
            sample_rng = torch.Generator().manual_seed(p['training_sample_seed_root']+mi)
            opt = torch.optim.Adam(model.fields.parameters(), lr=p['learning_rate'])
            schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=p['epochs'], eta_min=p['final_learning_rate'])
            ce, acc, best_logits = validate(model, x[va], y[va], p, mi)
            best_ce, best_epoch, best = ce, 0, copy.deepcopy(model.state_dict())
            history = [{'epoch': 0, 'validation_CE': ce, 'validation_accuracy': acc}]
            for epoch in range(p['epochs']):
                model.train()
                order = torch.randperm(len(tr), generator=order_rng)
                ids = torch.as_tensor(tr)[order]
                total, steps = 0., 0
                for first in range(0, len(ids), p['batch']):
                    ii = ids[first:first+p['batch']]
                    opt.zero_grad(set_to_none=True)
                    loss = F.cross_entropy(model(x[ii], 256 if method == 'binomial_mean_STE_T256' else 0, sample_rng), y[ii])
                    assert torch.isfinite(loss)
                    loss.backward()
                    assert all(torch.isfinite(v.grad).all() for v in model.fields)
                    opt.step()
                    with torch.no_grad():
                        for field in model.fields:
                            field.clamp_(-4, 4)
                    total += float(loss.detach())*len(ii)
                    steps += 1
                ce, acc, logits = validate(model, x[va], y[va], p, mi)
                item = {'model': mi, 'method': method, 'epoch': epoch+1, 'train_CE': total/len(tr),
                        'validation_CE': ce, 'validation_accuracy': acc, 'steps': steps,
                        'order_sha256': hashlib.sha256(order.numpy().tobytes()).hexdigest()}
                history.append(item)
                if ce < best_ce:
                    best_ce, best_epoch, best, best_logits = ce, epoch+1, copy.deepcopy(model.state_dict()), logits
                schedule.step()
                write(OUT/f'model{mi}_{method}_history.json', history)
                print(json.dumps({**item, 'elapsed_seconds': time.monotonic()-started}), flush=True)
            model.load_state_dict(best)
            exported = model.export()
            exported['selected_validation_logits'] = best_logits
            for name in ['conv1', 'conv2', 'head']:
                np.testing.assert_array_equal(exported[name+'_bias'], initial[name+'_bias'])
            path = OUT/f'model{mi}_{method}.npz'
            save(path, **exported)
            state['rows'].append({'model': mi, 'method': method, 'selected_epoch': best_epoch, 'validation_CE': best_ce,
                                  'path': str(path), 'sha256': sha(path), 'history_sha256': sha(OUT/f'model{mi}_{method}_history.json')})
            write(OUT/'manifest.json', state)
    guard(p)
    assert len(state['rows']) == 10
    state.update(complete=True, elapsed_seconds=time.monotonic()-started)
    write(OUT/'manifest.json', state)
    print('All10validation-selected probability fits complete; native confirmation remains open', flush=True)


if __name__ == '__main__':
    {'freeze': freeze, 'run': run}[sys.argv[1]]()
