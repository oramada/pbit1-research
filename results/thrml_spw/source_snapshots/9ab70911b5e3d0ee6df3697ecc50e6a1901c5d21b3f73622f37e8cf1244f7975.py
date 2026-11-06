"""All fixed contrasts for the exact-budget native CNN sampling allocation."""
import datetime
import json
from pathlib import Path
import sys
import numpy as np
from results.thrml_spw.fashion_cnn_native.cohort.common import ROOT, sha, write, guard, metrics

PARENT = ROOT/'allocated_native_evidence'
BASE = ROOT/'cohort_evidence'
OUT = ROOT/'allocated_statistics_evidence'
NAMES = ['allocated_minus_uniform', 'uniform_minus_FP32_F64', 'allocated_minus_FP32_F64']


def draws(values, baseline, bm, bs):
    sampled = [values[ci][bm[:, :, None], bs[ci]].mean((1, 2)) for ci in range(2)]
    fp32 = baseline[bm].mean(1)
    return np.stack([sampled[1]-sampled[0], sampled[0]-fp32, sampled[1]-fp32])


def prepare():
    OUT.mkdir(exist_ok=True)
    assert not (OUT/'protocol.json').exists()
    parent = json.loads((PARENT/'protocol.json').read_text())
    paths = [PARENT/'protocol.json', ROOT/'layer_allocation_evidence/completion.json', BASE/'statistics.json',
             BASE/'control_logits.npz', BASE/'resource_ledger.json', BASE/'test_data.npz']
    sources = list(Path(__file__).parent.glob('*.py'))+[ROOT/'cohort/common.py']
    p = {'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
         'source_sha256': {str(path): sha(path) for path in sources},
         'input_sha256': {str(path): sha(path) for path in paths}, 'contrasts': NAMES,
         'bootstrap_seed': parent['bootstrap_seed'], 'bootstrap_draws': 5000,
         'metric_order': ['accuracy_fraction', 'mean_CE_nats'],
         'scope': parent['statistics'], 'parent_pipeline_pid': 3893677, 'parent_pipeline_startticks': 46547427}
    guard(p)
    write(OUT/'protocol.json', p)


def run():
    p = json.loads((OUT/'protocol.json').read_text())
    guard(p)
    assert not (OUT/'statistics.json').exists()
    parent = json.loads((PARENT/'protocol.json').read_text())
    guard(parent)
    m = json.loads((PARENT/'manifest.json').read_text())
    audit = json.loads((PARENT/'native_audit.json').read_text())
    assert m['complete'] and audit['complete'] and audit['passed'] and audit['manifest_sha256'] == sha(PARENT/'manifest.json')
    assert m['protocol_sha256'] == audit['protocol_sha256'] == sha(PARENT/'protocol.json')
    expected = {(ci, mi, seed) for ci in range(2) for mi in range(5) for seed in range(10)}
    assert len(m['rows']) == len(audit['rows']) == 100
    assert {(r['condition'], r['model'], r['seed']) for r in m['rows']} == expected
    assert {(r['condition'], r['model'], r['seed']) for r in audit['rows']} == expected
    assert all(r['all_receipts_and_final_logits_exact'] for r in audit['rows'])
    recovered = {r['name']: r for r in audit['rows']}
    assert len(recovered) == 100
    for row in m['rows']:
        for key in ['sha256', 'model', 'condition', 'seed', 'budgets']:
            assert recovered[row['name']][key] == row[key]
    assert m['scheduled_updates'] == audit['scheduled_updates'] == 85248000000
    assert m['counter_entries'] == audit['counter_entries'] == 4718250000
    labels = np.load(BASE/'test_data.npz')['labels']
    controls = np.load(BASE/'control_logits.npz')
    baseline = np.stack([metrics(controls[f'c0_model{mi}'], labels) for mi in range(5)])
    values, rows = np.empty((2, 5, 10, 2)), []
    for row in m['rows']:
        assert sha(row['path']) == row['sha256']
        case = np.load(row['path'])
        assert case['logits'].shape == (10000, 10) and np.isfinite(case['logits']).all()
        v = metrics(case['logits'], labels)
        values[row['condition'], row['model'], row['seed']] = v
        rows.append({k: row[k] for k in ['name', 'condition', 'model', 'seed', 'budgets']} |
                    {'accuracy': float(v[0]), 'mean_CE': float(v[1]), 'clipped_coordinates': case['clipped_coordinates'].tolist()})
    rng = np.random.default_rng(p['bootstrap_seed'])
    bm = rng.integers(0, 5, size=(5000, 5))
    bs = rng.integers(0, 10, size=(2, 5000, 5, 10))
    distributions = draws(values, baseline, bm, bs)
    points = np.stack([(values[1]-values[0]).mean((0, 1)), (values[0]-baseline[:, None, :]).mean((0, 1)),
                       (values[1]-baseline[:, None, :]).mean((0, 1))])
    np.savez_compressed(OUT/'statistical_arrays.npz', values=values, baseline=baseline, bootstrap_models=bm,
                        bootstrap_seeds=bs, distributions=distributions)
    guard(p)
    write(OUT/'statistics.json', {'protocol_sha256': sha(OUT/'protocol.json'), 'manifest_sha256': sha(PARENT/'manifest.json'),
                                 'native_audit_sha256': sha(PARENT/'native_audit.json'),
                                 'statistical_arrays_sha256': sha(OUT/'statistical_arrays.npz'), 'rows': rows,
                                 'summaries': [{'condition': parent['conditions'][ci], 'mean': values[ci].mean((0, 1)).tolist(),
                                                'model_means': values[ci].mean(1).tolist()} for ci in range(2)],
                                 'baseline_mean': baseline.mean(0).tolist(),
                                 'contrasts': [{'contrast': name, 'difference': points[i].tolist(),
                                                'descriptive_95_interval': np.quantile(distributions[i], [.025, .975], axis=0).T.tolist()}
                                               for i, name in enumerate(NAMES)], 'scope': p['scope']})
    resource = json.loads((BASE/'resource_ledger.json').read_text())
    write(OUT/'resource_ledger.json', {'weight_pbits': 2664, 'activation_pbits': 0,
                                     'base_parameter_payload_bytes': resource['parameter_payload_bytes'],
                                     'schedule_bytes_per_model': {'uniform': 2, 'allocated': 6},
                                     'logical_payload_with_schedule_bytes': {'uniform': resource['parameter_payload_bytes']+2,
                                                                              'allocated': resource['parameter_payload_bytes']+6},
                                     'observed_updates_per_bank': 681984, 'warmup_updates_per_bank': 2664,
                                     'full_production_retained_sample_updates': m['scheduled_updates'],
                                     'full_public_retained_sample_updates': audit['scheduled_updates'],
                                     'full_warmup_updates_per_pass': 333000000,
                                     'preflight_native_retained_updates': json.loads((PARENT/'preflight.json').read_text())['native_updates'],
                                     'preflight_public_retained_updates': json.loads((PARENT/'preflight.json').read_text())['public_updates'],
                                     'preflight_warmup_updates_per_pass': 106560,
                                     'bank_counter_entries_int32': 37746, 'bank_counter_bytes': 150984,
                                     'model_layer_samples': parent['schedules'],
                                     'scope': 'Observed scheduled updates, warmup sweeps, metadata and observer counters are distinct. No measured physical flips/area/latency/energy or full software memory bound.'})
    print('All100matched-budget outcomes and3contrasts retained; independent checks pending', flush=True)


if __name__ == '__main__':
    {'prepare': prepare, 'run': run}[sys.argv[1]]()
