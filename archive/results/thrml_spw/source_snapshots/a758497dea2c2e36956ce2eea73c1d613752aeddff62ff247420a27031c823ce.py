"""Independent raw-score, fixed-threshold, paired-bootstrap and budget audit."""
import csv,json
from pathlib import Path
import numpy as np
from .fine_range_common import OUT,condition,sha,atomic_json
def rows(name):
    with (OUT/name).open() as f:return list(csv.DictReader(f))
def close(x,y):np.testing.assert_allclose(x,y,rtol=1e-12,atol=1e-12)
def bootstrap(delta,p):
    d=np.asarray(delta,float).reshape(5,4,8).mean(-1);rng=np.random.default_rng(p['bootstrap_seed']);si=rng.integers(0,5,(p['bootstrap_draws'],5));bi=rng.integers(0,4,(p['bootstrap_draws'],4));return np.array([d[si[k]][:,bi[k]].mean() for k in range(p['bootstrap_draws'])])
def main():
    p=json.loads((OUT/'protocol.json').read_text());m=json.loads((OUT/'manifest.json').read_text());assert m['native_model_cases']==10 and m['native_operators']==240 and not m['test_scores_used_for_selection'] and m['execution']['protocol_sha256']==sha(OUT/'protocol.json') and sha(OUT/'tokens.npz')==p['tokens_sha256']
    for name,h in m['execution']['source_sha256'].items():assert sha(name)==h
    truth=np.load(OUT/'tokens.npz')['test_tokens'][:,1:];base=np.load(OUT/'baseline_tokens.npz');baseline=rows('baseline_metrics.csv');assert len(baseline)==1 and baseline[0]['condition']=='FP32';ref=base['FP32_nll'].astype(float).mean(-1);refac=(base['FP32_predictions']==truth).mean(-1)
    for k,v in {'cross_entropy':ref.mean(),'perplexity':np.exp(ref.mean()),'token_accuracy':refac.mean()}.items():close(float(baseline[0][k]),v)
    native=rows('native_metrics.csv');summary=rows('summary.csv');draws=np.load(OUT/'quality_draws.npz');assert len(native)==10 and len(summary)==2;loss={};acc={};total=0;budgets={}
    for name in ['coarse','fine']:
        cfg=condition(p,name);ll=[];cc=[];budget=0
        for seed in range(5):
            folder=OUT/'runs'/f'{name}_seed{seed}';run=json.loads((folder/'run.json').read_text());assert run['execution']==m['execution'] and run['status']=='complete' and run['condition']==name and run['seed']==seed and run['exact_prefix_input_checks']==9600 and len(run['stages'])==24 and sha(folder/'prefix_tokens.npz')==run['token_archive_sha256'];raw=np.load(folder/'prefix_tokens.npz');np.testing.assert_array_equal(raw['prefix0_nll'],base['FP32_nll']);np.testing.assert_array_equal(raw['prefix0_predictions'],base['FP32_predictions']);updates=0
            for stage,s in enumerate(run['stages']):
                meta=json.loads(Path(s['metadata']).read_text());sig=meta['signature'];c=p['control_artifacts'][str(cfg['group'])][stage];assert s['stage']==sig['stage']==stage and sig['seed']==seed and sig['condition']==name and s['samples']==sig['samples']==cfg['samples'] and s['group']==sig['group']==cfg['group'] and s['inputs_sha256']==sig['inputs_sha256'] and sig['controls_sha256']==c['sha256']==sha(c['path']) and sig['execution_sha256']==sha(OUT/'execution.json') and sha(s['output'])==meta['output_sha256'];assert s['prior_native_output_sha256']==[sha(folder/f'stage{k}_native_output.npz') for k in range(stage)];assert meta['free_node_updates']==4*cfg['samples']*c['weights'];updates+=meta['free_node_updates']
            assert updates==run['free_node_updates']==4*42467328*cfg['samples'];budget+=updates;rr=next(r for r in native if r['condition']==name and int(r['seed'])==seed);assert int(rr['free_node_updates'])==updates;ce=float(raw['prefix24_nll'].astype(float).mean());ac=float(np.mean(raw['prefix24_predictions']==truth))
            for k,v in {'cross_entropy':ce,'perplexity':np.exp(ce),'token_accuracy':ac}.items():close(float(rr[k]),v);close(run['scores'][k],v)
            ll.append(raw['prefix24_nll'].astype(float).mean(-1));cc.append((raw['prefix24_predictions']==truth).mean(-1))
        loss[name],acc[name]=np.stack(ll),np.stack(cc);total+=budget;budgets[name]=budget
    assert total==m['native_observed_updates'] and budgets['coarse']==2*budgets['fine']
    for name in ['coarse','fine']:
        ce=bootstrap(loss[name]-ref,p);ac=bootstrap(acc[name]-refac,p);close(ce,draws[name+'_CE_vs_FP32']);close(ac,draws[name+'_accuracy_vs_FP32']);delta=float((loss[name]-ref).mean());ad=float((acc[name]-refac).mean());upper=float(np.quantile(ce,.975));lower=float(np.quantile(ac,.025));point=delta<=np.log(1.05) and ad>=-.01;passed=point and upper<=np.log(1.05) and lower>=-.01;row=next(r for r in summary if r['condition']==name);stored=next(r for r in m['summary'] if r['condition']==name)
        values={'cross_entropy':loss[name].mean(),'perplexity':np.exp(loss[name].mean()),'relative_perplexity':np.exp(delta),'token_accuracy':acc[name].mean(),'accuracy_difference':ad,'bootstrap_relative_upper':np.exp(upper),'bootstrap_accuracy_lower':lower}
        for k,v in values.items():close(float(row[k]),v);close(stored[k],v)
        assert (row['point_passed']=='True')==stored['point_passed']==point and (row['passes']=='True')==stored['passes']==passed
    ce=bootstrap(loss['fine']-loss['coarse'],p);ac=bootstrap(acc['fine']-acc['coarse'],p);close(ce,draws['paired_CE']);close(ac,draws['paired_accuracy']);pair=json.loads((OUT/'paired_summary.json').read_text());assert pair==m['paired_contrast'] and pair['contrast']=='fine-minus-coarse'
    for k,v in {'CE_difference':(loss['fine']-loss['coarse']).mean(),'perplexity_ratio':np.exp((loss['fine']-loss['coarse']).mean()),'accuracy_difference':(acc['fine']-acc['coarse']).mean(),'CE_sensitivity_low':np.quantile(ce,.025),'CE_sensitivity_high':np.quantile(ce,.975),'accuracy_sensitivity_low':np.quantile(ac,.025),'accuracy_sensitivity_high':np.quantile(ac,.975)}.items():close(pair[k],v)
    for name in ['reservation_audit.json','model_audit.json','native_audit.json']:assert json.loads((OUT/name).read_text())['passed']
    r={'passed':True,'native_model_cases':10,'native_operators':240,'all_60000_bootstrap_values_recomputed':True,'fine_observed_updates_half_coarse':True,'production_node_updates_verified':total,'FP32_baseline_verified_by_original_model_path_audit':True,'source_sha256':sha(__file__),'scope':'Independent score/resource/bootstrap audit after complete original-model/public-native path checks. Audit correctness does not imply either fixed quality gate passes; task failures remain reportable. One ideal checkpoint/short contexts,51.84% converted parameters, no physical advantage.'};atomic_json(OUT/'evidence_audit.json',r);print(json.dumps(r,indent=2))
if __name__=='__main__':main()
