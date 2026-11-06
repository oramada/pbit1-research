"""Conditional neural-error covariance across the five original training draws."""
from pathlib import Path
import argparse, hashlib, json, sys, time
import numpy as np
from .prepare_v1 import OUT, sha
from .lfm3_full_preflight_v1 import ROOT, point_hash
from .benchmark_v1 import atomic_json
sys.path.insert(0, str(Path('publish/pbit1/src').resolve()))
from pbit1 import FinitePBitProgram


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze-only', action='store_true')
    args = parser.parse_args()
    training = OUT / 'lfm3_training_depth_v1'
    result = json.loads((training / 'results.json').read_text())
    policies = json.loads((OUT / 'lfm3_calibration_v2/results.json').read_text())['policies']
    selection = OUT / 'lfm3_neuron_structure_v1/moments.npz'
    protocol = {
        'source_sha256': sha(__file__), 'training_result_sha256': sha(training / 'results.json'),
        'SDK_source_sha256': sha('publish/pbit1/src/pbit1/program.py'),
        'selected_channels_sha256': sha(selection),
        'seeds': [case['case']['seed'] for case in result['cases'][2:]],
        'groups': ['64 largest training-error channels per block', '64 fixed random channels per block'],
        'positions_per_block': 8224, 'samples': 32768,
        'scope': 'Average across input positions of the sample covariance across five independent model weight draws, conditional on each fixed input. All24blocks, two64channel subsets. Training-only descriptive statistic; channel selection already used this cohort. No heldout quality claim or full2048x2048covariance claim.'
    }
    root = OUT / 'lfm3_error_covariance_v2'
    root.mkdir(exist_ok=True)
    if (root / 'protocol.json').exists():
        assert json.loads((root / 'protocol.json').read_text()) == protocol
    else:
        atomic_json(root / 'protocol.json', protocol)
    if args.freeze_only:
        print(json.dumps({'status': 'five_draw_conditional_error_covariance_protocol_frozen', 'protocol_sha256': sha(root / 'protocol.json')}))
        return
    import fcntl
    lock = (OUT / 'gpu_research.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    import torch
    from transformers import AutoModelForCausalLM
    from safetensors import safe_open
    from .leaf_offload_v1 import LeafOffload
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    assert 'GB10' in torch.cuda.get_device_name()
    torch.cuda.set_per_process_memory_fraction(2 * 1024**3 / torch.cuda.get_device_properties(0).total_memory)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert not (root / 'failure.json').exists()
    if (root / 'results.json').exists():
        return
    begin = time.monotonic()
    try:
        with np.load(selection) as f:
            indices = np.stack([f['top64_error_channels'], f['random64_channels']])
        assert indices.shape == (2, 24, 64)
        with np.load(OUT / 'lfm3_data_v1/training_windows.npz') as f:
            inputs = f['input_ids'].copy()
        baseline = np.empty((2, 24, 8224, 64), np.float32)
        for row in range(32):
            with np.load(training / 'BF16' / f'layer_states{row:02d}.npz') as f:
                for layer in range(24):
                    value = (f[f'layer{layer:02d}'].astype(np.uint32) << 16).view(np.float32).reshape(257, 2048)
                    for group in range(2):
                        baseline[group, layer, row*257:(row+1)*257] = value[:, indices[group, layer]]
        sums = np.zeros_like(baseline, dtype=np.float64)
        cross = np.zeros((2, 24, 64, 64), np.float64)
        model = AutoModelForCausalLM.from_pretrained(ROOT, local_files_only=True, trust_remote_code=True, dtype=torch.bfloat16, attn_implementation='eager').cpu().eval()
        sys.modules[type(model.model.layers[0].conv).__module__].can_use_lfm3_conv = lambda *a: False
        params = dict(model.named_parameters())
        assert list(params) == [p['name'] for p in policies]
        capture = {}
        handles = [layer.register_forward_hook(lambda m, a, o, i=i: capture.__setitem__(i, o.detach().cpu().float().numpy())) for i, layer in enumerate(model.model.layers)]
        checked = []
        for seed in protocol['seeds']:
            original = training / f'selected_seed{seed}'
            old = json.loads((original / 'parameters.json').read_text())
            with torch.no_grad(), safe_open(str(ROOT / 'model.safetensors'), framework='pt', device='cpu') as weights:
                for index, policy in enumerate(policies):
                    raw = weights.get_tensor(policy['name'])
                    program = FinitePBitProgram.encode(raw.float().numpy(), group_size=policy['selected_group'], layout=policy['selected_layout'])
                    sample = program.reconstruct(32768, seed=seed, parameter_index=index)
                    value = torch.from_numpy(sample.weights).to(torch.bfloat16)
                    ref = old['records'][index]
                    assert sample.counts_sha256 == ref['counts_sha256']
                    assert hashlib.sha256(value.view(torch.uint16).numpy().tobytes()).hexdigest() == ref['BF16_sha256']
                    params[policy['name']].copy_(value)
                    del raw, program, sample, value
                    atomic_json(root / 'progress.json', {'seed': seed, 'stage': 'reconstruct', 'parameter': index, 'elapsed_seconds': time.monotonic()-begin})
            assert point_hash(model) == old['point_sha256']
            with torch.inference_mode(), LeafOffload(model, 'cuda') as offloader:
                for row, ids in enumerate(inputs):
                    capture.clear()
                    out = model(input_ids=torch.from_numpy(ids[None]).cuda(), use_cache=False)
                    loss = torch.nn.functional.cross_entropy(out.logits[0, :-1].float(), torch.from_numpy(ids[1:]).cuda(), reduction='none').cpu().double().numpy()
                    with np.load(original / f'row{row:02d}.npz') as f:
                        np.testing.assert_array_equal(loss, f['token_NLL'])
                    del out
                    for layer in range(24):
                        value = capture[layer].reshape(257, 2048)
                        for group in range(2):
                            error = value[:, indices[group, layer]].astype(np.float64) - baseline[group, layer, row*257:(row+1)*257]
                            sums[group, layer, row*257:(row+1)*257] += error
                            cross[group, layer] += error.T @ error
                    atomic_json(root / 'progress.json', {'seed': seed, 'stage': 'trace', 'row': row, 'elapsed_seconds': time.monotonic()-begin})
            del offloader
            checked.append({'seed': seed, 'weight_count_hashes_checked': 286, 'replayed_targets': 8192})
        covariance = np.empty_like(cross)
        mean_outer = np.empty_like(cross)
        summaries = []
        for group in range(2):
            for layer in range(24):
                position_sum = sums[group, layer]
                mean_outer[group, layer] = position_sum.T @ position_sum / 5
                matrix = (cross[group, layer] - mean_outer[group, layer]) / (8224 * 4)
                matrix = (matrix + matrix.T) / 2
                covariance[group, layer] = matrix
                eigen = np.linalg.eigvalsh(matrix)
                assert eigen.min() >= -1e-10 * max(eigen.max(), 1e-30)
                std = np.sqrt(np.maximum(np.diag(matrix), 0))
                corr = matrix / np.maximum(np.outer(std, std), 1e-30)
                summaries.append({'group': group, 'layer': layer, 'mean_absolute_error_correlation': float(np.mean(np.abs(corr[np.triu_indices(64, 1)]))),
                                  'covariance_participation_ratio': float(np.trace(matrix)**2 / np.sum(matrix*matrix)),
                                  'largest_eigenvalue_fraction': float(eigen[-1] / eigen.sum())})
        with (root / 'covariances.npz').open('xb') as f:
            np.savez_compressed(f, covariance=covariance, channels=indices, cross_sums=cross, mean_outer_products=mean_outer, positions=np.array(8224), draws=np.array(5))
        atomic_json(root / 'results.json', {'status': 'all24_blocks_five_draw_conditional_error_covariances_measured', 'protocol_sha256': sha(root / 'protocol.json'), 'checked': checked, 'summaries': summaries, 'scope': protocol['scope'], 'elapsed_seconds': time.monotonic()-begin})
    except Exception as error:
        atomic_json(root / 'failure.json', {'type': type(error).__name__, 'message': str(error)})
        raise

if __name__ == '__main__':
    main()
