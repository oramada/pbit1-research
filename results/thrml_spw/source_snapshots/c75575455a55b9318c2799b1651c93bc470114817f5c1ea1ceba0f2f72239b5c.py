"""Frozen conventions for the T4096 follow-up; earlier experiments stay immutable."""
import json
import numpy as np
from results.thrml_spw.cifar_cnn_native.probability_cohort.common import (
    ROOT, BASE, WORD, METHODS, sha, write, save, guard, seal, checkpoint, keys, metrics)

OUT = ROOT/'probability_high_budget_evidence'
PARENT = ROOT/'probability_cohort_evidence'
PRECISION = ROOT/'precision_evidence'
WITNESS = ROOT/'probability_high_budget_witness_evidence'
REFERENCES = ['BF16_weights_F64', 'BF16_Torch_CPU', 'FP32_F64', 'own_mean',
              'own_T1024', 'original_T1024', 'word_T256']
CONTRASTS = [(method, reference) for method in range(2) for reference in REFERENCES]+[(1, 'expected_T4096')]


def parent_closed():
    for folder, status in [(PARENT, 'full_probability_finetuned_RGB_independently_audited_goal_active'),
                           (PRECISION, 'FP16_BF16_RGB_controls_audited_goal_active')]:
        assert json.loads((folder/'pipeline.json').read_text())['status'] == status
        seal(folder/'completion.json')
    seal(PARENT/'review_completion.json')
    seal(WITNESS/'completion.json')
    seal(WITNESS/'review_completion.json')
    assert json.loads((WITNESS/'evidence_audit.json').read_text())['passed']


def references(parent, base, word, precision):
    return {'BF16_weights_F64': precision['control_metrics'][1],
            'BF16_Torch_CPU': precision['control_metrics'][3],
            'FP32_F64': base['baseline_metrics'][0],
            'own_mean': parent['mean_metrics'], 'own_T1024': parent['values'][:, 2],
            'original_T1024': base['values'][2], 'word_T256': word['values'][2]}


def compute(values, refs, bm, bs, parent_bs, base_bs, word_bs):
    assert values.shape == (2, 5, 10, 2)
    output = []
    for method, name in CONTRASTS:
        left = values[method]
        right = values[0] if name == 'expected_T4096' else refs[name]
        if name in ['own_mean', 'own_T1024']:
            right = right[method]
        dist = left[bm[:, :, None], bs[method]].mean((1, 2))
        if right.ndim == 2:
            dist -= right[bm].mean(1)
            point = left.mean((0, 1))-right.mean(0)
        else:
            seeds = {'own_T1024': parent_bs[method], 'original_T1024': base_bs,
                     'word_T256': word_bs, 'expected_T4096': bs[0]}[name]
            dist -= right[bm[:, :, None], seeds].mean((1, 2))
            point = left.mean((0, 1))-right.mean((0, 1))
        output.append({'method': method, 'reference': name, 'point': point, 'bootstrap': dist})
    return output
