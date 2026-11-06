"""Conditional five-seed quality of the prior-selected mean program."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from .adaptation_native_v2_statistics import summary


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path('results/thrml_spw/lfm3_1p7b'); out = root / 'counted_mean_t4096'
    p = json.loads((root / 'counted_mean_t4096_protocol.json').read_text())
    inputs, values, work = {}, [], []
    controls = None
    for seed in p['seeds']:
        path = out / 'mean' / str(seed) / 'independent_audit.json'
        a = json.loads(path.read_text()); inputs[str(path)] = sha(path)
        assert a['status'] == 'all_direct_program_coverage_losses_and_resource_receipts_verified'
        assert a['seed'] == seed and a['arm'] == 'mean' and a['targets'] == 2048
        result = path.with_name('results.json')
        assert a['results_sha256'] == sha(result); inputs[str(result)] = sha(result)
        current = {k: a['losses'][k]['mean_NLL_FP64'] for k in ['BF16', 'same_fit_BF16', 'nominal_controls']}
        if controls is None: controls = current
        assert current == controls
        values.append(a['losses']['native']['mean_NLL_FP64'])
        work.append({k: a[k] for k in ['retained_scheduled_evaluations', 'warmup_scheduled_evaluations', 'max_temporary_weight_count_bytes']})
    critical = float(p['MC_t4_critical'])
    absolute = summary(values, critical)
    gaps = {k: summary(np.asarray(values) - v, critical) for k, v in controls.items()}
    r = {'status': 'all_five_prior_selected_mean_cases_summarized', 'inputs_sha256': inputs,
        'protocol_sha256': sha(root / 'counted_mean_t4096_protocol.json'), 'controls_NLL': controls,
        'native_NLL': absolute, 'native_minus_control_NLL': gaps,
        'native_subset_perplexity': math.exp(absolute['mean']),
        'native_over_same_fit_BF16_PPL_ratio': math.exp(gaps['same_fit_BF16']['mean']),
        'native_over_same_fit_BF16_PPL_ratio_MC95_interval': [math.exp(v) for v in gaps['same_fit_BF16']['MC95_t4_interval']],
        'conditional_same_fit_BF16_gate_passed': gaps['same_fit_BF16']['MC95_t4_interval'][1] <= p['BF16_quality_gate_NLL_margin'],
        'work_per_seed': work, 'scope': p['scope']}
    with (out / 'statistics.json').open('x') as f: json.dump(r, f, indent=2)
    print(json.dumps({'status': r['status'], 'NLL': absolute, 'gate_passed': r['conditional_same_fit_BF16_gate_passed']}))


if __name__ == '__main__':
    main()
