"""Fixed-range probability learning; binomial means are a training surrogate."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

PREFIXES = ['conv1', 'conv2', 'head']


class ProbabilityCNN(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.fields = nn.ParameterList()
        self.shapes = []
        self.original_ranges = []
        for i, name in enumerate(PREFIXES):
            w = original[name+'_weight'].astype(float)
            flat = w.reshape(w.shape[0], -1)
            lo, hi = flat.min(1), flat.max(1)
            center, radius = (lo+hi)/2, (hi-lo)/2
            assert np.all(radius > 0)
            p = np.clip((flat-center[:, None])/(2*radius[:, None])+.5, 1/(1+np.exp(8.)), 1/(1+np.exp(-8.)))
            code = np.rint((np.log(p/(1-p))/2+4)*255/8).astype(np.uint8)
            field = (-4+code.astype(float)*8/255).astype(np.float32)
            self.fields.append(nn.Parameter(torch.from_numpy(field.copy())))
            self.register_buffer('center'+str(i), torch.tensor(center, dtype=torch.float32))
            self.register_buffer('radius'+str(i), torch.tensor(radius, dtype=torch.float32))
            self.register_buffer('bias'+str(i), torch.tensor(original[name+'_bias'].copy()))
            self.shapes.append(w.shape)
            self.original_ranges.append((center.copy(), radius.copy()))
        self.register_buffer('spans', torch.tensor(original['activation_span'], dtype=torch.float32))

    def probability(self, layer):
        field = self.fields[layer].clamp(-4, 4)
        quantized = -4+torch.round((field+4)*255/8)*8/255
        # Explicit biased straight-through derivative of field quantization.
        effective = field+(quantized-field).detach()
        return torch.sigmoid(2*effective)

    def weights(self, layer, samples, generator):
        p = self.probability(layer)
        c = getattr(self, 'center'+str(layer))[:, None]
        r = getattr(self, 'radius'+str(layer))[:, None]
        mean = c-r+2*r*p
        if samples:
            count = torch.binomial(torch.full_like(p, samples), p.detach(), generator=generator)
            sampled = c-r+2*r*count/samples
            # Forward is a binomial mean; backward follows expected weights.
            # This is not an unbiased Bernoulli gradient estimator.
            mean = mean+(sampled-mean).detach()
        return mean.reshape(self.shapes[layer])

    def quantize_activation(self, x, layer):
        bounded = torch.clamp(x/self.spans[layer], 0, 1)
        q = torch.round(255*bounded)/255
        return self.spans[layer]*(bounded+(q-bounded).detach())

    def forward(self, x, samples=0, generator=None):
        x = self.quantize_activation(x, 0)
        x = F.max_pool2d(F.relu(F.conv2d(x, self.weights(0, samples, generator), self.bias0, padding=1)), 2)
        x = self.quantize_activation(x, 1)
        x = F.max_pool2d(F.relu(F.conv2d(x, self.weights(1, samples, generator), self.bias1, padding=1)), 2)
        x = self.quantize_activation(x, 2).flatten(1)
        return F.linear(x, self.weights(2, samples, generator), self.bias2)

    def export(self):
        arrays = {}
        for i, prefix in enumerate(PREFIXES):
            field = self.fields[i].detach().numpy().astype(float)
            codes = np.rint((np.clip(field, -4, 4)+4)*255/8).astype(np.uint8)
            fields = (-4+codes.astype(float)*8/255).astype(np.float32)
            center, radius = self.original_ranges[i]
            expected = center[:, None]-radius[:, None]+2*radius[:, None]/(1+np.exp(-2*fields.astype(float)))
            arrays.update({prefix+'_latent_fields': field, prefix+'_commands': codes,
                           prefix+'_fields': fields, prefix+'_center': center, prefix+'_radius': radius,
                           prefix+'_weight': expected.reshape(self.shapes[i]),
                           prefix+'_bias': getattr(self, 'bias'+str(i)).numpy().copy()})
        arrays['activation_lower'] = np.zeros(3)
        arrays['activation_span'] = self.spans.numpy().astype(float)
        return arrays


def native_encoding(exported):
    """Preserve learned commands/fixed ranges; never re-fit min/max to the mean."""
    return [{'center': exported[p+'_center'], 'radius': exported[p+'_radius'],
             'fields': exported[p+'_fields'].T.copy(), 'commands': exported[p+'_commands'],
             'bias': exported[p+'_bias'].astype(float)} for p in PREFIXES]
