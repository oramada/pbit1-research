"""Identity control STE plus explicit projection; endpoints can move inward."""
import torch
from .probability_training import ProbabilityLinear


class ProjectedProbabilityLinear(ProbabilityLinear):
    def moments(self):
        _, programmed = self.commands()
        # Forward includes finite clipping and command quantization. Backward
        # deliberately uses identity through both. project() enforces the
        # feasible box after every optimizer step, including at endpoints.
        h = self.theta + (programmed.to(self.theta.dtype) - self.theta).detach()
        p = torch.sigmoid(2 * h)
        shape = (-1, self.theta.shape[-1])
        return ((self.lower + self.span * p).reshape(shape),
                (self.span.square() * p * (1 - p)).reshape(shape))
