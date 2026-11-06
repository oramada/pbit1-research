"""Independent task scores, paired sensitivity, resource counts and combined audit."""
import csv,json
from pathlib import Path
import numpy as np
from .verify_transformer_control import sha
OUT=Path('results/thrml_spw/transformer_calibration_floor');OLD=Path('results/thrml_spw/calibrated_transformer')


def read(name):
    with (OUT/name).open() as f:return list(csv.DictReader(f))


def main():
    p=json.loads((OUT/'protocol.json').read_text());m=json.loads((OUT/'manifest.json').read_text());assert m['config']==p and m['native_forward_runs']==5 and m['native_operator_runs']==60 and m['native_probe_runs']==0
    for n,h in p['source_sha256'].items():assert sha(n)==h
    assert sha(OLD/'manifest.json')==p['source_manifest_sha256'] and sha(OLD/'evidence_audit.json')==p['source_evidence_audit_sha256'];oldmanifest=json.loads((OLD/'manifest.json').read_text());assert not next(r for r in oldmanifest['profiles'] if r['condition']=='cal4096')['passes']
    tokens=np.load(p['tokens_path'])['tokens'];truth=tokens[:,1:];base=np.load(OLD/'baseline_tokens.npz');ref=float(base['FP32_nll'].astype(float).mean());refac=float(np.mean(base['FP32_predictions']==truth))
    with (OUT/'native_metrics.csv').open() as f:metrics=list(csv.DictReader(f))
    assert len(metrics)==5;updates=0
    for seed in range(5):
        folder=OUT/'runs'/f'oracle8_seed{seed}';r=json.loads((folder/'run.json').read_text());raw=np.load(folder/'prefix_tokens.npz');assert r['status']=='complete' and r['protocol_sha256']==sha(OUT/'protocol.json') and r['exact_prefix_input_checks']==64*78 and len(r['stages'])==12
        total=0
        for stage,s in enumerate(r['stages']):
            meta=json.loads(Path(s['metadata']).read_text());assert sha(s['inputs'])==meta['signature']['inputs_sha256'] and sha(s['output'])==meta['output_sha256'];assert s['prior_native_output_sha256']==[sha(folder/f'stage{k}_native_output.npy') for k in range(stage)]
            assert meta['signature']['condition']=='oracle8';total+=meta['free_node_updates']
        assert total==r['free_node_updates']==8*1024*21233664;updates+=total
        np.testing.assert_array_equal(raw['prefix0_nll'],base['FP32_nll']);np.testing.assert_array_equal(raw['prefix0_predictions'],base['FP32_predictions'])
        for stage in range(13):assert raw[f'prefix{stage}_nll'].shape==raw[f'prefix{stage}_predictions'].shape==(64,64) and np.isfinite(raw[f'prefix{stage}_nll']).all()
        ce=float(raw['prefix12_nll'].astype(float).mean());ac=float(np.mean(raw['prefix12_predictions']==truth));row=next(r for r in metrics if int(r['sampling_seed'])==seed)
        for k,v in [('cross_entropy',ce),('perplexity',float(np.exp(ce))),('token_accuracy',ac)]:assert r['scores'][k]==v;np.testing.assert_allclose(float(row[k]),v,atol=1e-14,rtol=1e-13)
    assert updates==m['native_free_evaluations']==5*8*1024*21233664
    nl={};acc={}
    for label,root,condition in [('measured8',OLD,'cal4096'),('ideal8',OLD,'ideal'),('oracle8',OUT,'oracle8')]:
        raw=[np.load(root/'runs'/f'{condition}_seed{s}'/'prefix_tokens.npz') for s in range(5)];nl[label]=np.stack([r['prefix12_nll'].astype(float).mean(-1) for r in raw]);acc[label]=np.stack([(r['prefix12_predictions']==truth).mean(-1) for r in raw])
        row=next(r for r in read('native_summary.csv') if r['condition']==label);ce=float(nl[label].mean());ac=float(acc[label].mean());expected={'cross_entropy':ce,'perplexity':float(np.exp(ce)),'relative_perplexity_FP32':float(np.exp(ce-ref)),'token_accuracy':ac,'accuracy_difference_FP32':ac-refac,'CE_seed_normal95_margin':float(1.96*nl[label].mean(-1).std(ddof=1)/np.sqrt(5))}
        for k,v in expected.items():np.testing.assert_allclose(float(row[k]),v,atol=1e-14,rtol=1e-13)
    draws=np.load(OUT/'paired_diagnostic_draws.npz')
    for label,left,right in [('oracle_minus_measured','oracle8','measured8'),('oracle_minus_ideal','oracle8','ideal8')]:
        delta=(nl[left]-nl[right]).reshape(5,8,8).mean(-1);adelta=(acc[left]-acc[right]).reshape(5,8,8).mean(-1)
        rg=np.random.default_rng(2026114701);si=rg.integers(0,5,(10000,5));bi=rg.integers(0,8,(10000,8))
        ce=np.array([delta[si[i]][:,bi[i]].mean() for i in range(10000)]);ac=np.array([adelta[si[i]][:,bi[i]].mean() for i in range(10000)])
        np.testing.assert_allclose(draws[label+'_CE'],ce,atol=1e-15,rtol=1e-13);np.testing.assert_allclose(draws[label+'_accuracy'],ac,atol=1e-15,rtol=1e-13)
        row=next(r for r in read('paired_diagnostics.csv') if r['contrast']==label);mean=float((nl[left]-nl[right]).mean());am=float((acc[left]-acc[right]).mean())
        for k,v in {'CE_difference':mean,'perplexity_ratio':float(np.exp(mean)),'accuracy_difference':am,'crossed_CE_low':np.quantile(ce,.025),'crossed_CE_high':np.quantile(ce,.975),'crossed_accuracy_low':np.quantile(ac,.025),'crossed_accuracy_high':np.quantile(ac,.975)}.items():np.testing.assert_allclose(float(row[k]),v,atol=1e-14,rtol=1e-13)
    for label in p['mean_conditions']:
        seeds=[0] if label=='ideal8' else range(5);raw=[np.load(OUT/'mean_runs'/f'{label}_seed{s}.npz') for s in seeds];ce=float(np.mean([r['nll'].astype(float).mean() for r in raw]));ac=float(np.mean([np.mean(r['predictions']==truth) for r in raw]));row=next(r for r in read('mean_summary.csv') if r['condition']==label)
        for k,v in {'cross_entropy':ce,'perplexity':float(np.exp(ce)),'relative_perplexity_FP32':float(np.exp(ce-ref)),'token_accuracy':ac}.items():np.testing.assert_allclose(float(row[k]),v,atol=1e-14,rtol=1e-13)
    diagnostics=json.loads((OUT/'weight_diagnostics.json').read_text());summary=json.loads((OUT/'weight_error_summary.json').read_text());n=sum(r['moments']['weights'] for r in diagnostics);assert n==summary['weights_across_device_profiles']==5*21233664
    for k,v in summary['components_mean_squared'].items():np.testing.assert_allclose(v,sum(r['moments'][k] for r in diagnostics)/n,atol=1e-18,rtol=1e-12)
    assert abs(sum(summary['components_mean_squared'][k] for k in ['measured_minus_known_SSE','oracle_grid_SSE','oracle_range_SSE','twice_estimate_grid','twice_estimate_range','twice_grid_range'])-summary['components_mean_squared']['total_SSE'])<1e-15
    convention=json.loads((OUT/'analysis_convention.json').read_text());assert not convention['quality_gates_applied'] and not convention['candidate_selected'] and convention['resampling_seed']==2026114701
    hooks=json.loads((OUT/'independent_hook_audit.json').read_text());assert hooks['passed'] and hooks['completed_runs_audited']==5
    mean=json.loads((OUT/'mean_evidence_audit.json').read_text());assert mean['passed'] and mean['mean_forward_runs']==16
    result={'passed':True,'native_forward_runs':5,'native_operator_runs':60,'native_probes':0,'mean_forward_runs':16,'paired_diagnostic_bootstrap_values':40000,'all_task_scores_pairing_statistics_and_resources_recomputed':True,'old_failed_measured_regime_retained':True,'source_sha256':sha(__file__),'manifest_sha256':sha(OUT/'manifest.json'),
        'scope':'Consumed-text controlled oracle/mean-weight diagnostics, not fresh confirmation or implementable calibration. Separate native-path audit required before combined evidence.'}
    (OUT/'task_score_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
    if (OUT/'native_path_audit.json').exists():combine()


def combine():
    task=json.loads((OUT/'task_score_audit.json').read_text());native=json.loads((OUT/'native_path_audit.json').read_text());mean=json.loads((OUT/'mean_evidence_audit.json').read_text());hooks=json.loads((OUT/'independent_hook_audit.json').read_text())
    assert task['passed'] and task['source_sha256']==sha(__file__) and task['manifest_sha256']==sha(OUT/'manifest.json') and native['passed'] and native['native_operator_runs']==60 and native['native_probe_runs']==0 and mean['passed'] and hooks['passed'] and hooks['completed_runs_audited']==5
    result={**task,'every_native_weight_node_and_model_path_independently_replayed':True,'native_path_audit_sha256':sha(OUT/'native_path_audit.json'),'mean_audit_sha256':sha(OUT/'mean_evidence_audit.json'),'hook_audit_sha256':sha(OUT/'independent_hook_audit.json')}
    (OUT/'evidence_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'combined_evidence_passed':True},indent=2))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--combine-only',action='store_true');args=parser.parse_args()
    if args.combine_only:combine()
    else:main()
