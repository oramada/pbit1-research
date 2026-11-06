"""Differentiable ACT12 operators with finite commands and optional IID counts.

FP64 affine weights are used within the operator, with BF16 rounding only at
the operator output. Binomial counts model independent repeated observations;
they are a Torch training surrogate, not a call to the THRML sampler. The
declared first-order STE differentiates the programmed expectation and treats
ACT12 quantization as identity. Native inference must validate trained programs.
"""
import types
import torch
import torch.nn.functional as F
from .probability_tensor import ProbabilityTensor
from .recomputed_probability import programmed_probability


class _Weights(torch.autograd.Function):
    @staticmethod
    def forward(ctx, theta, lower, span, group_shape, samples, generator, counts):
        ctx.save_for_backward(theta, span)
        ctx.group_shape = group_shape
        p = programmed_probability(theta, group_shape)
        if samples:
            if counts is None:
                counts = torch.binomial(torch.full_like(p, samples), p, generator=generator)
            else:
                counts = counts.reshape(group_shape).to(torch.float64)
                assert torch.isfinite(counts).all() and torch.equal(counts, counts.round())
                assert counts.min() >= 0 and counts.max() <= samples
            fraction = counts / samples
        else:
            assert counts is None
            fraction = p
        return (lower.to(torch.float64) + span.to(torch.float64) * fraction).reshape(theta.shape)

    @staticmethod
    def backward(ctx, gradient):
        theta, span = ctx.saved_tensors
        p = programmed_probability(theta, ctx.group_shape)
        value = gradient.to(torch.float64).reshape(ctx.group_shape) * span.to(torch.float64) * 2 * p * (1 - p)
        return value.reshape(theta.shape).to(theta.dtype), None, None, None, None, None, None


class OperatorProbabilityTensor(ProbabilityTensor):
    def weights(self, samples=0, generator=None, rows=None, counts=None):
        assert isinstance(samples, int) and samples >= 0
        assert rows is None or len(self.original_shape) == 2
        theta = self.theta if rows is None else self.theta.index_select(0, rows)
        lower = self.lower if rows is None else self.lower.index_select(0, rows)
        span = self.span if rows is None else self.span.index_select(0, rows)
        shape = self.group_shape if rows is None else (len(rows), *self.group_shape[1:])
        return _Weights.apply(theta, lower, span, shape, samples, generator, counts)


def quantize_act12(x):
    # Match the native worker's FP32 division and nearest-even rounding.
    xf = x.to(torch.float32)
    maximum = xf.detach().abs().max().item()
    scale = maximum / 2047 if maximum else 1.
    integer = (xf.detach() / scale).round().clamp(-2047, 2047).to(torch.int32)
    value = integer.to(torch.float64) * scale
    identity = x.to(torch.float64)
    return identity + (value - identity).detach(), integer, scale


def execute(kind, x, weight):
    """Worker-layout inputs, canonical weights; FP64 output before BF16 cast."""
    value, _, _ = quantize_act12(x)
    if kind == 'linear':
        return value @ weight.T
    if kind == 'conv':
        assert weight.ndim == 3 and weight.shape[1] == 1
        layout = weight[:, 0, :].T
    else:
        assert kind in ['embedding', 'elementwise']
        layout = weight.reshape(1, -1)
    return (value * layout[None]).sum(1)


def install(model, samples=0, seed=202610320):
    """Replace all weighted forwards, preserving one encoder for tied weights.

    Original frozen BF16 parameters remain references. The caller owns the
    returned mode dictionary and can switch samples/seed between forwards.
    No cache, packed documents, biases or segmented convolution supported.
    """
    targets = [(name, module, module._parameters['weight']) for name, module in model.named_modules()
               if module._parameters.get('weight') is not None]
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    state = {'samples': samples, 'generator': torch.Generator(device='cpu').manual_seed(seed), 'calls': []}
    encoders, names, aliases = {}, {}, {}

    def forward_for(kind):
        def forward(module, x):
            encoder = module.spw_encoder
            count = state['samples']
            generator = state['generator']
            if kind == 'embedding':
                rows, inverse = torch.unique(x, sorted=True, return_inverse=True)
                weight = encoder.weights(count, generator, rows=rows)
                # Embedding's constant input quantizes exactly to one.
                y = weight[inverse]
            elif kind == 'conv':
                assert module.padding == (6,) and module.kernel_size == (7,) and module.stride == (1,)
                assert module.dilation == (1,) and module.groups == x.shape[1]
                windows = F.pad(x, (6, 0)).unfold(-1, 7, 1)
                inputs = windows.permute(0, 2, 3, 1).reshape(-1, 7, x.shape[1])
                y = execute(kind, inputs, encoder.weights(count, generator))
                y = F.pad(y.reshape(x.shape[0], x.shape[2], x.shape[1]).transpose(1, 2), (0, 6))
            elif kind == 'linear':
                y = execute(kind, x.reshape(-1, x.shape[-1]), encoder.weights(count, generator))
                y = y.reshape(*x.shape[:-1], encoder.original_shape[0])
            else:
                xf = x.float()
                normalized = (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + module.variance_epsilon)).to(x.dtype)
                y = execute(kind, normalized.reshape(-1, 1, x.shape[-1]), encoder.weights(count, generator))
                y = y.reshape(x.shape)
            state['calls'].append(module.spw_name)
            return y.to(module.weight.dtype)
        return forward

    for name, module, original in targets:
        assert original.device.type == 'cpu'
        assert getattr(module, 'bias', None) is None
        key = id(original)
        if key not in encoders:
            encoders[key] = OperatorProbabilityTensor(original)
            names[key] = name + '.weight'
        kind = ('embedding' if isinstance(module, torch.nn.Embedding) else 'linear' if isinstance(module, torch.nn.Linear)
                else 'conv' if isinstance(module, torch.nn.Conv1d) else 'elementwise')
        if kind == 'elementwise':
            assert original.ndim == 1 and hasattr(module, 'variance_epsilon')
        module.add_module('spw_encoder', encoders[key])
        module.spw_name = name
        module.forward = types.MethodType(forward_for(kind), module)
        aliases[name] = names[key]
    return state, {names[key]: value for key, value in encoders.items()}, aliases
