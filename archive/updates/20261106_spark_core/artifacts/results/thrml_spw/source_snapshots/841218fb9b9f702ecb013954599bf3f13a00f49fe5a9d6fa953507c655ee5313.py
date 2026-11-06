"""BF16 controls for the exact ten already evaluated probability-finetuned fits."""
import datetime
import json
from pathlib import Path
import sys
import numpy as np
import torch
from torch.nn import functional as F
from results.thrml_spw.cifar_cnn_native.model import CifarCNN16
from results.thrml_spw.cifar_cnn_native.probability_cohort.common import (
    ROOT, METHODS, checkpoint, sha, write, save, guard, seal)

OUT = ROOT/'same_fit_bf16_evidence'
LOW = ROOT/'probability_cohort_evidence'
HIGH = ROOT/'probability_high_budget_evidence'
BASE = ROOT/'cohort_evidence'
NAMES = ['BF16_weights_F64', 'BF16_Torch_CPU', 'BF16_weights_ACT8_F64']
PREFIXES = ['conv1', 'conv2', 'head']


def closed():
    for folder in [LOW, HIGH]:
        for name in ['completion.json', 'review_completion.json']:
            seal(folder/name)
        assert json.loads((folder/'statistics_audit.json').read_text())['passed']


def output(x, model, name):
    # BF16 reference weights are the FP32 representation of the learned means.
    if name == 'BF16_Torch_CPU':
        net = CifarCNN16().eval()
        with torch.no_grad():
            for prefix in PREFIXES:
                layer = getattr(net, prefix)
                layer.weight.copy_(torch.as_tensor(model[prefix+'_weight'], dtype=torch.float32))
                layer.bias.copy_(torch.as_tensor(model[prefix+'_bias'], dtype=torch.float32))
            net = net.to(torch.bfloat16)
            result = net(torch.as_tensor(x, dtype=torch.bfloat16))
        assert result.dtype == torch.bfloat16
        return result.float().numpy().astype(float)
    h = torch.as_tensor(x, dtype=torch.float64)
    with torch.no_grad():
        for li, prefix in enumerate(PREFIXES):
            if name == 'BF16_weights_ACT8_F64':
                span = float(model['activation_span'][li])
                h = span*torch.round(255*torch.clamp(h/span, 0, 1))/255
            w = torch.as_tensor(model[prefix+'_weight'], dtype=torch.float32).to(torch.bfloat16).double()
            b = torch.as_tensor(model[prefix+'_bias'], dtype=torch.float64)
            h = F.max_pool2d(F.relu(F.conv2d(h, w, b, padding=1)), 2) if li < 2 else F.linear(h.flatten(1), w, b)
    return h.numpy()


def metric(z, labels):
    shifted = z-z.max(1, keepdims=True)
    loss = np.log(np.exp(shifted).sum(1))-shifted[np.arange(len(labels)), labels]
    return np.array([np.mean(z.argmax(1) == labels), np.mean(loss)])


def prepare():
    OUT.mkdir(exist_ok=True)
    assert not (OUT/'protocol.json').exists()
    closed()
    check = json.loads((OUT/'selfcheck.json').read_text())
    assert check['passed'] and check['validation_paths'] == 30
    for path, digest in check['source_sha256'].items():
        assert sha(path) == digest
    high = json.loads((HIGH/'protocol.json').read_text())
    sources = dict(high['source_sha256'])
    for path in [*Path(__file__).parent.glob('*.py'),
                 ROOT/'model.py', Path('results/thrml_spw/fashion_cnn_native/cohort/common.py')]:
        sources[str(path)] = sha(path)
        assert (ROOT.parent/'source_snapshots'/(sha(path)+'.py')).exists()
    inputs = dict(high['input_sha256'])
    for folder in [LOW, HIGH]:
        for name in ['completion.json', 'review_completion.json']:
            receipt = json.loads((folder/name).read_text())
            inputs[str(folder/name)] = sha(folder/name)
            inputs.update(receipt['artifact_sha256'])
    for path in [BASE/'test_data.npz', LOW/'statistical_arrays.npz', HIGH/'statistical_arrays.npz',
                 OUT/'selfcheck.json', ROOT/'probability_finetune_evidence/manifest.json']:
        inputs[str(path)] = sha(path)
    for row in json.loads((ROOT/'probability_finetune_evidence/manifest.json').read_text())['rows']:
        inputs[row['path']] = row['sha256']
    protocol = {'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'source_sha256': sources, 'input_sha256': inputs, 'controls': NAMES,
        'methods': METHODS, 'samples': [64, 256, 1024, 4096], 'models': 5,
        'inference_seeds': 10, 'images': 10000, 'batch': 128, 'control_paths': 30,
        'contrast_count': 24, 'bootstrap_draws': 5000, 'new_native_updates': 0,
        'scope': 'Retrospective same-fit BF16 comparison on all ten original probability-finetuned exports; no optimization, new stochastic inference, favorable-method selection or fresh-data confirmation. Both methods/all four budgets/all three controls retained. BF16 weights use explicit FP32 cast of learned F64 means then BF16 nearest-even rounding. Weight-only controls retain deterministic original biases and F64 readout, with/without original ACT8 spans; whole CPU BF16 converts inputs/weights/biases and uses the unquantized-activation Torch module graph. CPU internal accumulation widths unclaimed. Parent same-fit ACT8 F64 means remain independently verified controls. Parent model/MC bootstrap indices reused per budget; fixed historically consumed images, unadjusted descriptive intervals, no formal equivalence margin/test. This removes unequal training history within each same-fit contrast, not model-selection bias or every arithmetic difference. No physical efficiency or per-weight numerical precision claim.'}
    guard(protocol)
    write(OUT/'protocol.json', protocol)
    print('Thirty same-fit BF16 paths and 24 contrasts frozen before new BF16 task scoring')


def evaluate():
    p = json.loads((OUT/'protocol.json').read_text())
    guard(p)
    closed()
    assert not (OUT/'evaluation.json').exists() and not (OUT/'logits.npz').exists()
    data = np.load(BASE/'test_data.npz'); x, y = data['inputs'], data['labels']
    logits = {}; controls = np.empty((2, 3, 5, 2))
    for method in range(2):
        for model_id in range(5):
            model = checkpoint(model_id, method)
            for ci, name in enumerate(NAMES):
                key = f'method{method}_c{ci}_model{model_id}'
                z = np.concatenate([output(x[first:first+p['batch']], model, name)
                                    for first in range(0, len(x), p['batch'])])
                assert z.shape == (10000, 10) and np.isfinite(z).all()
                logits[key] = z; controls[method, ci, model_id] = metric(z, y)
                print('Evaluated', key, flush=True)
    save(OUT/'logits.npz', **logits)
    low = np.load(LOW/'statistical_arrays.npz'); high = np.load(HIGH/'statistical_arrays.npz')
    base = np.load(BASE/'statistical_arrays.npz'); bm = base['bootstrap_models']
    distributions = {}; contrasts = []
    for bi, budget in enumerate(p['samples']):
        values = low['values'][:, bi] if bi < 3 else high['values']
        seeds = low['bootstrap_method_seeds'] if bi < 3 else high['bootstrap_method_seeds']
        for method in range(2):
            for ci, name in enumerate(NAMES):
                delta = values[method]-controls[method, ci, :, None]
                dist = delta[bm[:, :, None], seeds[method]].mean((1, 2))
                key = f'method{method}_budget{bi}_c{ci}'
                distributions[key] = dist
                contrasts.append({'method_index': method, 'method': METHODS[method], 'T': budget,
                    'control_index': ci, 'control': name, 'array_key': key,
                    'difference': delta.mean((0, 1)).tolist(),
                    'descriptive_95_interval': np.quantile(dist, [.025, .975], axis=0).T.tolist()})
    save(OUT/'statistical_arrays.npz', control_metrics=controls, **distributions)
    write(OUT/'statistics.json', {'protocol_sha256': sha(OUT/'protocol.json'),
        'controls': [{'method': METHODS[m], 'method_index': m, 'control': name,
            'control_index': ci, 'mean': controls[m, ci].mean(0).tolist()}
            for m in range(2) for ci, name in enumerate(NAMES)],
        'contrasts': contrasts, 'metric_order': ['accuracy_fraction', 'mean_CE_nats'],
        'logits_sha256': sha(OUT/'logits.npz'),
        'statistical_arrays_sha256': sha(OUT/'statistical_arrays.npz'), 'scope': p['scope']})
    guard(p)
    write(OUT/'evaluation.json', {'complete': True, 'control_paths': 30,
        'protocol_sha256': sha(OUT/'protocol.json'), 'logits_sha256': sha(OUT/'logits.npz'),
        'statistics_sha256': sha(OUT/'statistics.json'), 'new_native_updates': 0})


if __name__ == '__main__':
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    {'prepare': prepare, 'evaluate': evaluate}[sys.argv[1]]()
