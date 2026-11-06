"""Paired consumed-text diagnostic comparisons; no new confirmation decisions."""
import json,csv
from pathlib import Path
import numpy as np
from .fine_range_common import sha,atomic_json,atomic_npz
from .confirmation_statistics import bootstrap_paired_difference
OUT=Path('results/thrml_spw/fine_range_quantization_controls')
def main():
    p=json.loads((OUT/'protocol.json').read_text());assert json.loads((OUT/'evidence_audit.json').read_text())['passed'];source=Path(p['source_cohort']);tokens=np.load(p['source_tokens'])['test_tokens'];truth=tokens[:,1:];det=np.load(OUT/'raw_tokens.npz');native={};draws={};rows=[]
    for name in ['coarse','fine']:
        rr=[np.load(source/'runs'/f'{name}_seed{s}/prefix_tokens.npz') for s in range(5)];native[name]=(np.stack([r['prefix24_nll'].astype(float).mean(-1) for r in rr]),np.stack([(r['prefix24_predictions']==truth).mean(-1) for r in rr]))
    for v in p['variants']:
        names=['coarse','fine'] if v['kind']=='FP32' else ['fine' if v['group']==16 else 'coarse'];ref=det[v['name']+'_nll'].astype(float).mean(-1);refac=(det[v['name']+'_predictions']==truth).mean(-1)
        for name in names:
            loss,acc=native[name];ce=bootstrap_paired_difference(loss,np.broadcast_to(ref,loss.shape),seed=2026118200);ac=bootstrap_paired_difference(acc,np.broadcast_to(refac,acc.shape),seed=2026118200);key=name+'_minus_'+v['name'];draws[key+'_CE']=ce;draws[key+'_accuracy']=ac;delta=float((loss-ref).mean());rows.append({'native':name,'control':v['name'],'CE_native_minus_control':delta,'perplexity_native_over_control':float(np.exp(delta)),'accuracy_native_minus_control':float((acc-refac).mean()),'CE_sensitivity_low':float(np.quantile(ce,.025)),'CE_sensitivity_high':float(np.quantile(ce,.975)),'accuracy_sensitivity_low':float(np.quantile(ac,.025)),'accuracy_sensitivity_high':float(np.quantile(ac,.975))})
    assert len(rows)==22 and len(draws)==44;atomic_json(OUT/'paired_comparisons.json',rows);atomic_npz(OUT/'paired_draws.npz',**draws);atomic_json(OUT/'comparison_manifest.json',{'comparisons':22,'resampling_values':440000,'source_sha256':sha(__file__),'draws_sha256':sha(OUT/'paired_draws.npz'),'paired_sha256':sha(OUT/'paired_comparisons.json'),'scope':'Consumed-text diagnostic crossed seed/bank sensitivity. Deterministic references broadcast across seeds, not five independent control runs. No prior gate changed.'});print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
