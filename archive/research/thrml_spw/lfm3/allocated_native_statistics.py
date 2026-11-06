"""Frozen, paired comparison of allocated and uniform mean-trained programs."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path('results/thrml_spw/lfm3_1p7b')
    out = root / 'allocated_native'
    protocol = json.loads((root / 'allocated_native_protocol.json').read_text())
    inputs, values, work = {}, {}, {}
    critical = float(protocol['MC_t4_critical'])
    def read(path):
        inputs[str(path)] = sha(path)
        return json.loads(path.read_text())
    def summarize(x):
        x = np.asarray(x, dtype=np.float64)
        mean = float(x.mean()); se = float(x.std(ddof=1) / math.sqrt(5))
        return {'values': x.tolist(), 'mean': mean, 'standard_error': se,
                'MC95_t4_interval': [mean - critical * se, mean + critical * se]}
    controls = None
    weights_hashes = None
    for method, folder in [('allocated', 'allocated_native'), ('uniform', 'adaptation_native_v2')]:
        values[method], work[method] = [], []
        for seed in protocol['seeds']:
            case = root / folder / 'mean' / str(seed)
            audit = read(case / 'independent_audit.json')
            result = read(case / 'results.json')
            assert audit['status'] == 'all_direct_program_coverage_losses_and_resource_receipts_verified'
            assert audit['results_sha256'] == sha(case / 'results.json')
            assert audit['targets'] == 2048 and audit['seed'] == seed and audit['arm'] == 'mean'
            assert result['package_sha256'] == protocol['mean_program_package_sha256']
            current = {s: audit['losses'][s]['mean_NLL_FP64'] for s in ['BF16', 'same_fit_BF16', 'nominal_controls']}
            if controls is None:
                controls, weights_hashes = current, result['same_fit_BF16_weight_hashes']
            assert controls == current and weights_hashes == result['same_fit_BF16_weight_hashes']
            values[method].append(audit['losses']['native']['mean_NLL_FP64'])
            work[method].append({k: audit[k] for k in ['retained_scheduled_evaluations', 'warmup_scheduled_evaluations']})
    arms = {}
    for method in values:
        absolute = summarize(values[method])
        gap = summarize(np.asarray(values[method]) - controls['same_fit_BF16'])
        arms[method] = {'native_NLL': absolute, 'native_minus_same_fit_BF16_NLL': gap,
            'native_subset_perplexity': math.exp(absolute['mean']),
            'native_over_same_fit_BF16_PPL_ratio': math.exp(gap['mean']),
            'conditional_BF16_quality_gate_passed': gap['MC95_t4_interval'][1] <= protocol['BF16_quality_gate_NLL_margin']}
    for a, u in zip(work['allocated'], work['uniform'], strict=True):
        assert a['retained_scheduled_evaluations'] <= u['retained_scheduled_evaluations']
        assert a['warmup_scheduled_evaluations'] == u['warmup_scheduled_evaluations']
    report = {'status': 'all_five_paired_allocated_uniform_cases_summarized', 'arms': arms,
        'controls_NLL': controls, 'paired_allocated_minus_uniform_NLL': summarize(np.asarray(values['allocated']) - values['uniform']),
        'work_per_seed': work, 'inputs_sha256': inputs, 'protocol_sha256': sha(root / 'allocated_native_protocol.json'),
        'serial_operator_sample_cycles': protocol['serial_operator_sample_cycles'], 'scope': protocol['scope']}
    with (out / 'statistics.json').open('x') as f:
        json.dump(report, f, indent=2)
    print(json.dumps({'status': report['status'], 'arms': arms}))


if __name__ == '__main__':
    main()
