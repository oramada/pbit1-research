"""First-order projected STE with recomputed probabilities in backward.

No FP64 probability tensors are saved for the mean-weight path. Existing FP32
controls and fixed ranges are referenced; temporary backward workspace remains.
This is a Torch surrogate, not a gradient through native THRML sampling.
"""
import torch
from .probability_tensor import ProbabilityTensor


def programmed_probability(theta, group_shape):
    h = theta.detach().to(torch.float64).clamp(-4, 4)
    codes = torch.round((h + 4) * 4095 / 8).clamp(0, 4095).to(torch.int32)
    fields = codes.to(torch.float32) * (8 / 4095) - 4
    return torch.sigmoid(2 * fields.to(torch.float64)).reshape(group_shape)


class _Moments(torch.autograd.Function):
    @staticmethod
    def forward(ctx, theta, lower, span, group_shape, output_dtype, include_variance):
        ctx.save_for_backward(theta, span)
        ctx.group_shape = group_shape
        ctx.set_materialize_grads(False)
        p = programmed_probability(theta, group_shape)
        mean = (lower.to(torch.float64) + span.to(torch.float64) * p).reshape(theta.shape).to(output_dtype)
        variance = (span.to(torch.float64).square() * p * (1 - p)).reshape(theta.shape) if include_variance else None
        return mean, variance

    @staticmethod
    def backward(ctx, grad_mean, grad_variance):
        theta, span = ctx.saved_tensors
        p = programmed_probability(theta, ctx.group_shape)
        derivative = 2 * p * (1 - p)
        width = span.to(torch.float64)
        gradient = None
        if grad_mean is not None:
            gradient = grad_mean.to(torch.float64).reshape(ctx.group_shape) * width * derivative
        if grad_variance is not None:
            variance_gradient = grad_variance.to(torch.float64).reshape(ctx.group_shape) * width.square() * (1 - 2 * p) * derivative
            gradient = variance_gradient if gradient is None else gradient + variance_gradient
        return (None if gradient is None else gradient.reshape(theta.shape).to(theta.dtype)), None, None, None, None, None


class _VarianceMean(torch.autograd.Function):
    @staticmethod
    def forward(ctx, theta, span, group_shape):
        ctx.save_for_backward(theta, span)
        ctx.group_shape = group_shape
        p = programmed_probability(theta, group_shape)
        return (span.to(torch.float64).square() * p * (1 - p)).mean()

    @staticmethod
    def backward(ctx, gradient):
        theta, span = ctx.saved_tensors
        p = programmed_probability(theta, ctx.group_shape)
        value = gradient * span.to(torch.float64).square() * 2 * p * (1 - p) * (1 - 2 * p) / theta.numel()
        return value.reshape(theta.shape).to(theta.dtype), None, None


class RecomputedProbabilityTensor(ProbabilityTensor):
    def moments(self, rows=None, include_variance=True):
        assert rows is None or len(self.original_shape) == 2
        theta = self.theta if rows is None else self.theta.index_select(0, rows)
        lower = self.lower if rows is None else self.lower.index_select(0, rows)
        span = self.span if rows is None else self.span.index_select(0, rows)
        shape = self.group_shape if rows is None else (len(rows), *self.group_shape[1:])
        return _Moments.apply(theta, lower, span, shape, torch.float64, include_variance)

    def forward(self, original_weight):
        self.forward_calls = getattr(self, 'forward_calls', 0) + 1
        return _Moments.apply(self.theta, self.lower, self.span, self.group_shape, original_weight.dtype, False)[0]

    def variance_mean(self):
        return _VarianceMean.apply(self.theta, self.span, self.group_shape)
