"""Independent prospective text reservation and all-free finite-control audit."""
import json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from huggingface_hub.constants import HF_HUB_CACHE
from transformers import AutoTokenizer
from .verify_transformer_control import sha


def main():
    out=Path('results/thrml_spw/calibrated_transformer');p=json.loads((out/'protocol.json').read_text());assert sha(out/'tokens.npz')==p['tokens_sha256']
    used=[]
    for a in p['excluded_archives']:
        assert sha(a['path'])==a['sha256'];used.extend(np.load(a['path'])[a['key']].tolist())
    source=Path(hf_hub_download('Salesforce/wikitext','wikitext-2-raw-v1/test-00000-of-00001.parquet',repo_type='dataset',revision=p['dataset_revision'],local_files_only=True));assert sha(source)==p['source_parquet_sha256']
    tokenizer=AutoTokenizer.from_pretrained(Path(HF_HUB_CACHE)/f"models--distilbert--distilgpt2/snapshots/{p['revision']}",local_files_only=True)
    texts=pq.read_table(source,columns=['text']).column('text').to_pylist();ids=np.asarray(tokenizer('\n'.join(t for t in texts if t and t.strip()),add_special_tokens=False)['input_ids'])
    legacy=np.sort(np.random.default_rng(17).choice(len(ids)-64,4,replace=False));assert legacy.tolist()==p['legacy_excluded_offsets'];used=np.array(used+legacy.tolist())
    eligible=np.arange(len(ids)//65)*65;eligible=eligible[~np.any((eligible[:,None]<used[None]+65)&(used[None]<eligible[:,None]+65),axis=1)]
    selected=np.sort(np.random.default_rng(2026114700).choice(eligible,64,replace=False));saved=np.load(out/'tokens.npz')
    np.testing.assert_array_equal(saved['token_offsets'],selected);np.testing.assert_array_equal(saved['tokens'],np.stack([ids[i:i+65] for i in selected]));assert saved['tokens'].shape==(64,65)
    assert not np.any((selected[:,None]<used[None]+65)&(used[None]<selected[:,None]+65))
    assert sha(p['master_weights_path'])==p['master_weights_sha256'];master=np.load(p['master_weights_path']);n=0
    for stage in range(12):
        c=p['control_artifacts'][stage];assert sha(c['path'])==c['sha256'];a=np.load(c['path']);w=master[f'weight_{stage}'];bias=master[f'bias_{stage}'];assert set(a.files)=={'codes','lower','span','bias'} and a['codes'].dtype==np.uint8
        low=np.stack([w[i:i+128].min(0) for i in range(0,len(w),128)]);high=np.stack([w[i:i+128].max(0) for i in range(0,len(w),128)]);span=high-low
        np.testing.assert_array_equal(a['lower'],low);np.testing.assert_array_equal(a['span'],span);np.testing.assert_array_equal(a['bias'],bias)
        lo=np.repeat(low,128,0)[:len(w)];sp=np.repeat(span,128,0)[:len(w)];q=np.zeros_like(w);np.divide(w-lo,sp,out=q,where=sp!=0)
        eps=1/(1+np.exp(8));q=np.clip(q.astype(float),eps,1-eps);assert np.all((q>0)&(q<1))
        b=((np.log(q)-np.log1p(-q))/2).astype(np.float32);codes=np.rint((np.clip(b.astype(float),-4,4)+4)*255/8).astype(np.uint8);np.testing.assert_array_equal(a['codes'],codes);n+=w.size
    assert n==p['converted_weights']==21233664 and p['windows']==64 and p['targets']==4096 and p['model_scores_computed']==False
    for source,h in p['source_sha256'].items():assert sha(source)==h
    result={'passed':True,'new_disjoint_test_windows':64,'new_targets':4096,'converted_weights':n,'all12_controls_recomputed':True,'all_weights_have_finite_free_probability_targets':True,'model_scores_computed':False,'source_sha256':sha(__file__),'scope':'Pinned source/positions and original target-to-finite-control arithmetic checked before native model scoring. New positions from same corpus; does not establish task quality.'}
    (out/'preparation_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
