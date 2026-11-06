"""Gaussian negative-control conventions; no claim of native p-bit execution."""
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT = Path('results/thrml_spw/fashion_cnn_native')
BASE = ROOT/'cohort_evidence'
TRAIN = ROOT/'training'
OUT = ROOT/'gaussian_control_evidence'
MODES = ['original_FP32_mean', 'nominal_programmed_mean']
PREFIXES = ['conv1', 'conv2', 'head']
KEY_ROOT = 1677520


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2)+'\n')
    temp.replace(path)


def guard(p):
    for mapping in [p['source_sha256'], p['input_sha256']]:
        for path, h in mapping.items():
            assert sha(path) == h, path


def moments(model, prefix):
    weight = model[prefix+'_weight'].astype(float)
    flat = weight.reshape(len(weight), -1)
    center = (flat.max(1)+flat.min(1))/2
    radius = (flat.max(1)-flat.min(1))/2
    assert np.all(radius > 0)
    p = (flat-center[:, None])/(2*radius[:, None])+.5
    e = 1/(1+np.exp(8))
    p = np.clip(p, e, 1-e)
    commands = np.rint((np.log(p/(1-p))/2+4)*255/8).astype(np.uint8)
    fields = (-4+commands.astype(float)*8/255).astype(np.float32).astype(float)
    p = 1/(1+np.exp(-2*fields))
    mean = center[:, None]-radius[:, None]+2*radius[:, None]*p
    variance = 4*radius[:, None]**2*p*(1-p)
    return mean.reshape(weight.shape), variance.reshape(weight.shape), p, commands


def weights_for(model, mi, samples, seed, layer, first, count, mode):
    prefix = PREFIXES[layer]
    nominal, variance, _, _ = moments(model, prefix)
    mean = model[prefix+'_weight'].astype(float) if mode == 0 else nominal
    scale = np.sqrt(variance/samples)
    result = []
    for bank in range(first, first+count):
        # Deliberately shared normal variates across the two centering controls.
        rng = np.random.default_rng(np.random.SeedSequence([KEY_ROOT, mi, samples, seed, bank, layer]))
        result.append(mean+scale*rng.standard_normal(mean.shape))
    return np.stack(result)


def torch_forward(x, model, weights):
    import torch
    from torch.nn import functional as F
    h = torch.as_tensor(x, dtype=torch.float64)
    stages, codes, clipping = [], [], []
    banks, images = h.shape[:2]
    for li, prefix in enumerate(PREFIXES):
        span = float(model['activation_span'][li])
        clipping.append(int(((h < 0) | (h > span)).sum()))
        q = torch.round(255*torch.clamp(h/span, 0, 1))
        codes.append(q.numpy().astype(np.uint8))
        h = span*q/255
        w = torch.as_tensor(weights[li], dtype=torch.float64)
        b = torch.as_tensor(model[prefix+'_bias'].astype(float), dtype=torch.float64)
        if li < 2:
            _, _, channels, height, width = h.shape
            packed = h.permute(1, 0, 2, 3, 4).reshape(images, banks*channels, height, width)
            out = F.conv2d(packed, w.reshape(banks*w.shape[1], *w.shape[2:]), b.repeat(banks), padding=1, groups=banks)
            h = out.reshape(images, banks, w.shape[1], height, width).permute(1, 0, 2, 3, 4)
            h = F.max_pool2d(F.relu(h.reshape(banks*images, w.shape[1], height, width)), 2)
            h = h.reshape(banks, images, w.shape[1], height//2, width//2)
        else:
            h = torch.bmm(h.reshape(banks, images, -1), w.transpose(1, 2))+b
        stages.append(h.numpy())
    return h.numpy(), stages, codes, np.array(clipping, np.int64)


def metrics(logits, labels):
    maximum = logits.max(1)
    losses = maximum+np.log(np.exp(logits-maximum[:, None]).sum(1))-logits[np.arange(len(labels)), labels]
    return np.array([(logits.argmax(1) == labels).mean(), losses.mean()])
