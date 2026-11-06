"""Bounded numeric/gradient/export checks before any fine-tuning is executed."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from results.thrml_spw.cifar_cnn_native.probability_finetune.model import ProbabilityCNN, native_encoding

ROOT = Path('results/thrml_spw/cifar_cnn_native')
OUT = ROOT/'probability_finetune_evidence'


def main():
    OUT.mkdir(exist_ok=True)
    assert not (OUT/'selfcheck.json').exists()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    training = json.loads((ROOT/'training/manifest.json').read_text())
    inputs = torch.from_numpy(np.load(ROOT/'training/validation.npz')['inputs'][:32])
    checks = []
    for mi, row in enumerate(training['rows']):
        original = np.load(row['path'])
        model = ProbabilityCNN(original)
        assert sum(v.numel() for v in model.fields) == 3928
        for li in range(3):
            p = model.probability(li)
            expected_grad = 4*getattr(model, 'radius'+str(li))[:, None]*p.detach()*(1-p.detach())
            for samples in [0, 256]:
                model.zero_grad()
                model.weights(li, samples, torch.Generator().manual_seed(1807520+mi)).sum().backward()
                torch.testing.assert_close(model.fields[li].grad, expected_grad, rtol=3e-6, atol=3e-7)
        # Independent explicit functional graph; no calls to model.forward or
        # its activation/weight helpers in the reference graph.
        ww = []
        for li in range(3):
            field = model.fields[li].detach().clamp(-4, 4)
            effective = -4+torch.round((field+4)*255/8)*8/255
            prob = torch.sigmoid(2*effective)
            c = getattr(model, 'center'+str(li))[:, None]
            r = getattr(model, 'radius'+str(li))[:, None]
            ww.append((c-r+2*r*prob).reshape(model.shapes[li]))
        h = inputs
        for li in range(3):
            h = model.spans[li]*(torch.round(255*torch.clamp(h/model.spans[li], 0, 1))/255)
            if li < 2:
                h = F.max_pool2d(torch.clamp_min(F.conv2d(h, ww[li], getattr(model, 'bias'+str(li)), padding=1), 0), 2)
            else:
                h = F.linear(h.flatten(1), ww[li], model.bias2)
        with torch.no_grad():
            torch.testing.assert_close(model(inputs), h, rtol=0, atol=0)
        exported = model.export()
        encoded = native_encoding(exported)
        for li, name in enumerate(['conv1', 'conv2', 'head']):
            e = encoded[li]
            np.testing.assert_array_equal(e['fields'].T, exported[name+'_fields'])
            np.testing.assert_array_equal(e['commands'], exported[name+'_commands'])
            w = original[name+'_weight'].astype(float).reshape(original[name+'_weight'].shape[0], -1)
            np.testing.assert_array_equal(e['center'], (w.max(1)+w.min(1))/2)
            np.testing.assert_array_equal(e['radius'], (w.max(1)-w.min(1))/2)
            assert np.all((e['commands'] >= 0) & (e['commands'] <= 255))
            np.testing.assert_array_equal(e['bias'], original[name+'_bias'])
        checks.append({'model': mi, 'functional_mean_graph_exact': True, 'surrogate_gradient_checks': 6,
                       'fixed_range_native_export_layers_checked': 3})
    output = {'passed': True, 'rows': checks, 'new_native_updates': 0,
              'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__), Path(__file__).parent/'model.py', Path('results/thrml_spw/fashion_cnn_native/probability_finetune/model.py')]},
              'input_sha256': {str(ROOT/'training/manifest.json'): hashlib.sha256((ROOT/'training/manifest.json').read_bytes()).hexdigest(), str(ROOT/'training/validation.npz'): hashlib.sha256((ROOT/'training/validation.npz').read_bytes()).hexdigest()},
              'weight_fields_per_model': 3928,
              'scope': 'Only declared biased gradient/forward/export conventions checked; not unbiased stochastic gradients, trained-model validation or official native inference.'}
    (OUT/'selfcheck.json').write_text(json.dumps(output, indent=2)+'\n')
    print('FiveRGBmodels/3928fields each,30surrogate gradients/15fixed-range exports checked')


if __name__ == '__main__':
    main()
