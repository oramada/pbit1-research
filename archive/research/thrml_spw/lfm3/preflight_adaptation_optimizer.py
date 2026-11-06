"""Check chunked FP32 multi-step Adam audit arithmetic against Torch Adam."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch


def main():
    torch.set_num_threads(2)
    generator = torch.Generator().manual_seed(202610324)
    initial = torch.linspace(-4, 4, 1024)
    initial[:4] = torch.tensor([-4., 4., 0., 1e-10])
    actual = torch.nn.Parameter(initial.clone())
    optimizer = torch.optim.Adam([actual], lr=.0001, betas=(.9, .999), eps=1e-8, foreach=False)
    replay = initial.clone()
    m, v = torch.zeros_like(replay), torch.zeros_like(replay)
    records = []
    for step in range(1, 17):
        gradient = torch.randn(1024, generator=generator)
        gradient[:4] = torch.tensor([1., -1., 0., 1e-12])
        gradient *= 10. ** ((step % 7) - 3)
        actual.grad = gradient.clone()
        optimizer.step()
        with torch.no_grad():
            actual.clamp_(-4, 4)
        for start in range(0, len(initial), 128):
            end = start + 128
            g, first, second = gradient[start:end], m[start:end], v[start:end]
            first.lerp_(g, 1 - .9)
            second.mul_(.999).addcmul_(g, g, value=1 - .999)
            denominator = (second.sqrt() / math.sqrt(1 - .999 ** step)).add_(1e-8)
            replay[start:end].addcdiv_(first, denominator, value=-.0001 / (1 - .9 ** step))
            replay[start:end].clamp_(-4, 4)
        assert torch.equal(actual, replay)
        assert torch.equal(optimizer.state[actual]['exp_avg'], m)
        assert torch.equal(optimizer.state[actual]['exp_avg_sq'], v)
        records.append({'step': step, 'theta_bitwise_equal': True, 'moments_bitwise_equal': True})
    output = Path('results/thrml_spw/lfm3_1p7b/adaptation_optimizer_preflight.json')
    result = {'status': 'all16_chunked_FP32_Adam_steps_and_moments_bitwise_match_Torch',
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'torch_version': torch.__version__,
              'coordinates': len(initial), 'steps': records,
              'scope': 'Arithmetic preflight for independent gradient replay; synthetic gradients including zeros, tiny values, scale changes and projected boundaries. Not full-model optimization audit or quality evidence.'}
    with output.open('x') as f:
        json.dump(result, f, indent=2)
    print(json.dumps({'status': result['status'], 'coordinates': len(initial), 'steps': len(records)}))


if __name__ == '__main__':
    main()
