"""Actual model runner for an explicit ordered list of affine projections.

Evaluation and causal input checking reuse the audited first-block runner. Its
historical default target list is left unchanged.
"""
import numpy as np
from .transformer_block_gpu import BlockRunner


def projection_targets(blocks):
    return [f'transformer.h.{b}.{suffix}' for b in blocks for suffix in
            ('attn.c_attn','attn.c_proj','mlp.c_fc','mlp.c_proj')]


class SequentialRunner(BlockRunner):
    def __init__(self,model,targets):
        self.model=model
        self.targets=list(targets)
        if len(set(self.targets))!=len(self.targets):raise ValueError('Duplicate target modules')
        self.modules=[model.get_submodule(name) for name in self.targets]
        self.forwards=[m.forward for m in self.modules]
        self.original_weights=[m.weight.detach().clone() for m in self.modules]
        self.weights=[m.weight.detach().numpy().copy() if m.__class__.__name__=='Conv1D' else m.weight.detach().numpy().T.copy() for m in self.modules]
        self.biases=[m.bias.detach().numpy().copy() for m in self.modules]
        if any(w.ndim!=2 or not np.isfinite(w).all() for w in self.weights):raise ValueError('Finite affine matrices required')
