"""All LFM3 tensor shapes: fixed ranges, FP32 controls, projected identity STE.

Parametrization forward materializes BF16 mean weights for a Torch surrogate.
It does not perform native stochastic inference or ACT12 operator accumulation.
Move this module between devices without casting its FP32 controls/metadata.
"""
import hashlib
import math
import numpy as np
import torch


class ProbabilityTensor(torch.nn.Module):
    def __init__(self, weight):
        super().__init__()
        assert weight.ndim in [1, 2, 3] and weight.device.type != 'meta'
        self.original_shape = tuple(weight.shape)
        rows = weight.shape[0] if weight.ndim > 1 else 1
        width = math.prod(weight.shape[1:]) if weight.ndim > 1 else weight.numel()
        self.group_size = min(128, width)
        assert width % self.group_size == 0
        self.group_shape = (rows, width // self.group_size, self.group_size)
        w = weight.detach().cpu().float().numpy().astype(np.float64).reshape(self.group_shape)
        lower = w.min(2, keepdims=True)
        span = w.max(2, keepdims=True) - lower
        p = np.divide(w - lower, span, out=np.full_like(w, .5), where=span != 0).clip(1e-12, 1 - 1e-12)
        h = ((np.log(p) - np.log1p(-p)) / 2).clip(-4, 4)
        codes = np.rint((h + 4) * 4095 / 8).astype(np.uint16)
        self.initial_codes_sha256 = hashlib.sha256(codes.astype('<u2').tobytes()).hexdigest()
        fields = codes.astype(np.float32) * np.float32(8 / 4095) - np.float32(4)
        self.theta = torch.nn.Parameter(torch.from_numpy(fields.reshape(self.original_shape).copy()))
        self.register_buffer('lower', torch.from_numpy(lower.astype(np.float32)))
        self.register_buffer('span', torch.from_numpy(span.astype(np.float32)))

    def commands(self, rows=None):
        theta = self.theta.detach() if rows is None else self.theta.detach().index_select(0, rows)
        h = theta.to(torch.float64).clamp(-4, 4)
        codes = torch.round((h + 4) * 4095 / 8).clamp(0, 4095).to(torch.int32)
        fields = codes.to(torch.float32) * (8 / 4095) - 4
        return codes, fields

    def moments(self, rows=None, include_variance=True):
        assert rows is None or len(self.original_shape) == 2
        theta = self.theta if rows is None else self.theta.index_select(0, rows)
        lo = self.lower if rows is None else self.lower.index_select(0, rows)
        span = self.span if rows is None else self.span.index_select(0, rows)
        _, fields = self.commands(rows)
        h = theta.to(torch.float64)
        surrogate = h + (fields.to(torch.float64) - h).detach()
        shape = self.group_shape if rows is None else (len(rows), *self.group_shape[1:])
        p = torch.sigmoid(2 * surrogate.reshape(shape))
        output_shape = self.original_shape if rows is None else (len(rows), self.original_shape[1])
        mean = (lo.to(torch.float64) + span.to(torch.float64) * p).reshape(output_shape)
        variance = (span.to(torch.float64).square() * p * (1 - p)).reshape(output_shape) if include_variance else None
        return mean, variance

    def forward(self, original_weight):
        self.forward_calls = getattr(self, 'forward_calls', 0) + 1
        mean, _ = self.moments(include_variance=False)
        return mean.to(original_weight.dtype)

    def project(self):
        with torch.no_grad():
            self.theta.clamp_(-4, 4)

    def fingerprints(self):
        codes, _ = self.commands()
        encoded = codes.cpu().numpy().astype('<u2')
        return {'codes_sha256': hashlib.sha256(encoded.tobytes()).hexdigest(),
                'initial_codes_sha256': self.initial_codes_sha256,
                'lower_sha256': hashlib.sha256(self.lower.detach().cpu().numpy().astype('<f4').tobytes()).hexdigest(),
                'span_sha256': hashlib.sha256(self.span.detach().cpu().numpy().astype('<f4').tobytes()).hexdigest()}
