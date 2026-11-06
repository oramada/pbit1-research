"""Independent probability/accuracy arithmetic and complete-data audits."""
from pathlib import Path
import argparse,json,math
import numpy as np
from scipy.special import logsumexp
from .prepare_v1 import OUT,sha
from .causal_v1 import digest

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--case');args=parser.parse_args()
    root=OUT/'distilbert_benchmark_v1';protocol=json.loads((root/'protocol.json').read_text());p=protocol['plan'];assert sha(Path(__file__).with_name('distilbert_benchmark_v1.py'))==protocol['source_sha256']
    assert sha(p['arrays'])==p['arrays_sha256']
    with np.load(p['arrays'])as f:ids=f['input_ids'].copy();masks=f['attention_mask'].copy();labels=f['labels'].copy()
    cases=[c for c in p['cases']if c['name']==args.case]if args.case else p['cases'];assert cases
    dest=OUT/'distilbert_benchmark_audit_v1';dest.mkdir(exist_ok=True);reports=[]
    for case in cases:
        folder=root/case['name'];record=json.loads((folder/'results.json').read_text());point=json.loads((folder/'parameters.json').read_text());totals={'CE':0.,'correct':0,'Brier':0.,'reference_to_variant_KL':0.};count=0
        for index,start in enumerate(range(0,25000,p['batch_size'])):
            stop=min(25000,start+p['batch_size']);path=folder/f'batch{index:04d}.npz';receipt=json.loads((folder/f'batch{index:04d}.json').read_text())
            assert receipt['point_sha256']==point['point_sha256']and receipt['arrays_sha256']==sha(path)
            assert receipt['input_sha256']==digest(ids[start:stop])and receipt['mask_sha256']==digest(masks[start:stop])
            with np.load(path)as a:
                z=a['logits'].astype(np.float64);target=a['labels'];assert z.shape==(stop-start,2)and np.array_equal(target,labels[start:stop])
                logp=z-logsumexp(z,axis=1,keepdims=True);prob=np.exp(logp);ce=-logp[np.arange(stop-start),target];correct=(z.argmax(1)==target).astype(np.int8);brier=((prob-np.eye(2)[target])**2).sum(1)
                if case['kind']=='original':kl=np.zeros(stop-start)
                else:
                    with np.load(root/'BF16'/f'batch{index:04d}.npz')as b:ref=b['logits'].astype(np.float64)
                    logr=ref-logsumexp(ref,axis=1,keepdims=True);kl=np.sum(np.exp(logr)*(logr-logp),axis=1)
                for key,value in [('CE',ce),('correct',correct),('Brier',brier),('reference_to_variant_KL',kl)]:
                    np.testing.assert_allclose(value,a[key],atol=2e-12,rtol=1e-12);totals[key]+=float(value.sum())
            count+=stop-start
        assert count==25000
        means={k:v/count for k,v in totals.items()}
        for k in means:assert abs(means[k]-record['means'][k])<1e-12
        rows=point['parameters'];ledger=None
        if rows:
            assert len(rows)==104 and sum(r['parameters']for r in rows)==66955010
            for r in rows:
                expected='storage_flat'if case['kind']=='matched_flat'or len(r['shape'])!=2 else p['preferences'].get(r['name'],'storage_flat')
                assert r['layout']==expected and r['samples']==32768 and r['group']==128
            obs=sum(r['parameters']*r['samples']for r in rows);variance=sum(r['variance_of_SSE']for r in rows);error=sum(r['SSE_about_finite_control_mean_before_FP32_BF16_rounding']-r['expected_SSE']for r in rows)
            assert abs(error/math.sqrt(variance))<8;ledger={'logical_pbits':66955010,'observations':obs}
        reports.append({'case':case,'means':means,'ledger':ledger,'source_results_sha256':sha(folder/'results.json')})
        output=dest/(case['name']+'.json');output.write_text(json.dumps(reports[-1],indent=2)+'\n')
    if args.case:
        print(json.dumps(reports[0]));return
    r=json.loads((root/'results.json').read_text());assert len(reports)==12 and sha(root/'protocol.json')==r['protocol_sha256'];baseline=reports[0]['means'];stats={}
    for kind in ['selected','matched_flat']:
        group=[x['means']for x in reports if x['case']['kind']==kind];assert len(group)==5
        decline=np.array([baseline['correct']-x['correct']for x in group]);loss=np.array([x['CE']-baseline['CE']for x in group]);t=2.7764451051977987
        ua=float(decline.mean()+t*decline.std(ddof=1)/np.sqrt(5));ul=float(loss.mean()+t*loss.std(ddof=1)/np.sqrt(5))
        stats[kind]={'accuracy_decline_upper_MC95':ua,'accuracy_pass':ua<=.02,'categorical_perplexity_ratio':float(np.exp(loss.mean())),
            'categorical_perplexity_ratio_upper_MC95':math.exp(ul),'probability_pass':ul<=math.log(1.02),'joint_pass':ua<=.02 and ul<=math.log(1.02)}
        for k,v in stats[kind].items():
            if isinstance(v,bool):assert v==r['statistics'][kind][k]
            else:assert abs(v-r['statistics'][kind][k])<1e-12
    (dest/'results.json').write_text(json.dumps({'status':'all12cases_full25000_review_accuracy_probability_and_ledger_audits_passed','source_sha256':sha(__file__),
        'raw_result_sha256':sha(root/'results.json'),'cases':reports,'statistics':stats,'scope':'Recomputedalllogits-derivedmetrics; independentneuralreplay separate.'},indent=2)+'\n');print(json.dumps(stats,indent=2))

if __name__=='__main__':main()
