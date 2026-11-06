"""Independent Fisher-matrix design and fresh-position reservation audit."""
from pathlib import Path
import json
import numpy as np
import pyarrow.parquet as pq
from huggingface_hub.constants import HF_HUB_CACHE
from transformers import AutoTokenizer
from .verify_transformer_control import sha
OUT=Path('results/thrml_spw/probe_placement')


def main():
    p=json.loads((OUT/'protocol.json').read_text());ds=json.loads((OUT/'placement_designs.json').read_text());assert sha(OUT/'placement_designs.json')==p['placement_designs_sha256'] and len(ds)==12
    for n,h in p['source_sha256'].items():assert sha(n)==h
    for stage,c in enumerate(p['control_artifacts']):
        assert sha(c['path'])==c['sha256'];a=np.load(c['path']);b=(-4+a['codes'].astype(float)*8/255).astype(np.float32).astype(float);sp=np.repeat(a['span'],128,0).astype(float);q=1/(1+np.exp(-2*b));s=(2*sp*q*(1-q))**2
        a0,a1,a2=float(s.sum()),float((s*b).sum()),float((s*b*b).sum());row=ds[stage];assert row['stage']==stage
        for k,v in {'N':b.size,'A0':a0,'A1':a1,'A2':a2}.items():np.testing.assert_allclose(row['moments'][k],v,atol=1e-12,rtol=1e-12)
        scores=[]
        for low in range(128):
            probes=(-4+np.array([low,255-low])*8/255).astype(np.float32).astype(float);qq=1/(1+np.exp(-2*probes))
            info=sum(4*2048*pr*(1-pr)*np.outer([x,1],[x,1]) for x,pr in zip(probes,qq));cov=np.linalg.inv(info);score=float((cov[0,0]*a2+2*cov[0,1]*a1+cov[1,1]*a0)/b.size)
            r=row['candidates'][low];assert r['low_code']==low and r['high_code']==255-low;np.testing.assert_allclose(r['nominal_first_order_weight_MSE'],score,atol=1e-15,rtol=1e-11);scores.append(score)
        best=min(range(128),key=lambda i:(scores[i],i));assert row['chosen']['low_code']==best and row['fixed_baseline']['low_code']==96
    source=Path(p['source_parquet_path']);assert sha(source)==p['source_parquet_sha256'];texts=pq.read_table(source,columns=['text']).column('text').to_pylist()
    tokenizer=AutoTokenizer.from_pretrained(Path(HF_HUB_CACHE)/f"models--distilbert--distilgpt2/snapshots/{p['revision']}",local_files_only=True);ids=np.asarray(tokenizer('\n'.join(t for t in texts if t and t.strip()),add_special_tokens=False)['input_ids'])
    used=[]
    for e in p['excluded_archives']:
        assert sha(e['path'])==e['sha256'];used.extend(np.load(e['path'])[e['key']].tolist())
    used=np.array(used+p['legacy_excluded_offsets']);available=np.arange(len(ids)//65)*65;available=available[~np.any((available[:,None]<used[None]+65)&(used[None]<available[:,None]+65),axis=1)]
    chosen=np.sort(np.random.default_rng(2026116700).choice(available,64,replace=False));data=np.load(OUT/'tokens.npz');assert sha(OUT/'tokens.npz')==p['tokens_sha256'];np.testing.assert_array_equal(data['token_offsets'],chosen);np.testing.assert_array_equal(data['tokens'],np.stack([ids[i:i+65] for i in chosen]))
    assert not np.any((chosen[:,None]<used[None]+65)&(used[None]<chosen[:,None]+65)) and not p['model_scores_computed'] and not p['probe_counts_collected']
    r={'passed':True,'stage_designs':12,'candidate_scores_checked_with_independent_inverse_Fisher':1536,'new_disjoint_windows':64,'new_targets':4096,'no_new_profile_truth_or_task_scores_in_placement':True,'source_sha256':sha(__file__),'scope':'Before sampling/scoring, independently checks nominal weight-only Fisher criterion, choices and positional exclusions. Does not prove improved actual device/calibration/task quality.'}
    (OUT/'preparation_audit.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))


if __name__=='__main__':main()
