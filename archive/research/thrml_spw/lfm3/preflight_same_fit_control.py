"""Check temporary same-fit weights against a separately assigned BF16 model."""
import copy
import hashlib
import json
from pathlib import Path
import torch
from .probability_tensor import ProbabilityTensor


class Norm(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.linspace(.8, 1.2, 128, dtype=torch.bfloat16))
        self.variance_epsilon = 1e-6

    def forward(self, x):
        xf = x.float()
        return self.weight * (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + self.variance_epsilon)).to(x.dtype)


class Fixture(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(32, 128, padding_idx=31, dtype=torch.bfloat16)
        self.norm = Norm()
        self.conv = torch.nn.Conv1d(128, 128, 7, padding=6, groups=128, bias=False, dtype=torch.bfloat16)
        self.projection = torch.nn.Linear(128, 128, bias=False, dtype=torch.bfloat16)
        self.head = torch.nn.Linear(128, 32, bias=False, dtype=torch.bfloat16)
        self.head.weight = self.embedding.weight

    def forward(self, ids):
        x = self.norm(self.embedding(ids))
        x = self.conv(x.transpose(1, 2))[..., :ids.shape[1]].transpose(1, 2)
        return self.head(self.projection(x))


def main():
    torch.set_num_threads(2)
    torch.manual_seed(202610325)
    original = Fixture().eval()
    reference = copy.deepcopy(original)
    ids = torch.tensor([[0, 2, 31, 2, 15], [15, 3, 2, 0, 1]])
    targets = [(n, m) for n, m in original.named_modules() if m._parameters.get('weight') is not None]
    cache, saved, snapshots, aliases = {}, {}, {}, {}
    with torch.no_grad():
        raw = original(ids)
        for name, module in targets:
            key = id(module.weight)
            aliases[name] = key
            if key not in cache:
                encoder = ProbabilityTensor(module.weight)
                encoder.theta.add_(torch.linspace(-.3, .3, encoder.theta.numel()).reshape(encoder.theta.shape))
                encoder.project()
                cache[key] = encoder.moments(include_variance=False)[0].to(torch.bfloat16)
                saved[key] = module.weight.detach()
                snapshots[key] = module.weight.detach().clone()
            reference.get_submodule(name).weight.copy_(cache[key])
        expected = reference(ids)
    flag = {'same_fit': False}
    handles = []
    for name, module in targets:
        key = aliases[name]
        def pre(m, args, key=key):
            if flag['same_fit']:
                assert m.weight.data_ptr() == saved[key].data_ptr()
                m.weight.data = cache[key]
        def post(m, args, out, key=key):
            if flag['same_fit']:
                assert m.weight.data_ptr() == cache[key].data_ptr()
                m.weight.data = saved[key]
            return out
        handles.extend([module.register_forward_pre_hook(pre), module.register_forward_hook(post)])
    with torch.inference_mode():
        for _ in range(2):
            flag['same_fit'] = True
            assert torch.equal(original(ids), expected)
            assert all(torch.equal(m.weight, snapshots[aliases[name]]) for name, m in targets)
            flag['same_fit'] = False
            assert torch.equal(original(ids), raw)
    for handle in handles:
        handle.remove()
    assert original.embedding.weight is original.head.weight
    assert reference.embedding.weight is reference.head.weight
    assert not torch.equal(raw, expected)
    report = {'status': 'same_fit_original_operators_bitwise_match_assigned_BF16_reference_and_restore_weights',
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'weighted_modules': len(targets),
        'unique_weight_tensors': len(cache), 'shared_embedding_head': True, 'padding_index_tested': 31,
        'repetitions': 2, 'all_original_weights_bitwise_restored': True,
        'scope': 'CPU BF16 integration fixture with actual embedding/tiedhead, norm, convolution and linear operators. Temporary data swaps match a separately weight-assigned reference model exactly; originals restored between stages. No full LFM3/GPU or native quality claim.'}
    path = Path('results/thrml_spw/lfm3_1p7b/same_fit_control_preflight.json')
    with path.open('x') as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report))


if __name__ == '__main__':
    main()
