"""Fixed-range expected-operator training with explicit biased quantization STE.

This Torch surrogate is not a gradient through the THRML sampler. Native
inference must separately evaluate exported finite-field commands.
"""
import torch


class ProbabilityLinear(torch.nn.Module):
    def __init__(self, weights):
        super().__init__()
        w = weights.detach().to(torch.float64)
        assert w.ndim == 2 and w.shape[0] % 128 == 0
        grouped = w.reshape(-1, 128, w.shape[1])
        lower = grouped.amin(1, keepdim=True)
        span = grouped.amax(1, keepdim=True) - lower
        p = torch.where(span != 0, (grouped - lower) / torch.where(span == 0, torch.ones_like(span), span), .5)
        p = p.clamp(1e-12, 1 - 1e-12)
        self.theta = torch.nn.Parameter(((p.log() - torch.log1p(-p)) / 2).clamp(-4, 4))
        self.register_buffer('lower', lower)
        self.register_buffer('span', span)

    def commands(self):
        h = self.theta.detach().clamp(-4, 4)
        codes = torch.round((h + 4) * 4095 / 8).clamp(0, 4095).to(torch.int32)
        fields = codes.to(torch.float32) * (8 / 4095) - 4
        return codes, fields

    def moments(self):
        _, programmed = self.commands()
        h = self.theta.clamp(-4, 4)
        # Forward uses the actual decoded command; backward uses dh/dtheta=1
        # inside bounds. This intentionally differs from rounding's derivative.
        surrogate = h + (programmed.to(h.dtype) - h).detach()
        p = torch.sigmoid(2 * surrogate)
        shape = (-1, self.theta.shape[-1])
        mean = (self.lower + self.span * p).reshape(shape)
        variance = (self.span.square() * p * (1 - p)).reshape(shape)
        return mean, variance

    @staticmethod
    def quantize_inputs(x):
        xf = x.to(torch.float32)
        maximum = xf.detach().abs().max().item()
        scale = maximum / 2047 if maximum else 1.0
        integers = torch.round(xf / scale).clamp(-2047, 2047).to(torch.int32)
        values = integers.to(torch.float64) * scale
        return integers, scale, values

    def forward(self, x, samples):
        _, _, inputs = self.quantize_inputs(x)
        mean, variance = self.moments()
        return inputs @ mean, inputs.square() @ variance / samples

    def project(self):
        with torch.no_grad():
            self.theta.clamp_(-4, 4)
