"""Full real-data batch geometry and no-model observation arithmetic fixtures."""
from pathlib import Path
import hashlib,json
import numpy as np
from .r1_io_v1 import BASE,sha
from .r1_model_v1 import batches,observations
from .verify_r1_case_v1 import independent_batches,independent_observations


def main():
    mirror=BASE/'additional_assets_v1_mirror'/BASE/'additional_assets_v1';counts={};selected=[]
    for model in ['gpt2','distilbert']:
        spec={'model_type':model,'data':{'kind':'full_task','arrays':str(mirror/(model+'_tokens.npz')),
            'layout':str(mirror/(model+'_layout.json'))}}
        rows=targets=0;units=set()
        for a,b in zip(batches(spec),independent_batches(spec),strict=True):
            record,ids,mask,labels=a;rr,ii,mm,ll=b
            assert record==rr and ids.dtype==ii.dtype==np.int64
            np.testing.assert_array_equal(ids,ii)
            if mask is not None:np.testing.assert_array_equal(mask,mm)
            if labels is not None:np.testing.assert_array_equal(labels,ll)
            if model=='gpt2':
                n=record['target_stop']-record['target_start'];targets+=n;units.add(record['article'])
                assert 1<=record['target_start']-record['context_start']<ids.shape[1]<=1024
                if rows in [0,1]:selected.append((record,ids))
            else:
                assert ids.shape[1]==256 and len(labels)==len(ids)
                assert record['review_start']==targets;targets+=len(ids)
            rows+=1
        assert targets==(283225 if model=='gpt2' else 25000)
        if model=='gpt2':assert len(units)==62
        counts[model]={'batches':rows,'scored_targets_or_reviews':targets}
    rng=np.random.default_rng(202611970)
    for record,ids in selected:
        ids=ids%17;logits=rng.normal(size=(*ids.shape,17)).astype(np.float32)
        a=observations(record,ids,None,logits);b=independent_observations(record,ids,None,logits)
        for key in a:np.testing.assert_allclose(a[key],b[key],rtol=0,atol=1e-12)
    ids=np.zeros((3,4),np.int64);labels=np.asarray([0,1,1],np.int64)
    logits=np.asarray([[0,0],[1,2],[2,1]],np.float32);record={'task':'distilbert'}
    a=observations(record,ids,labels,logits);b=independent_observations(record,ids,labels,logits)
    for key in a:np.testing.assert_array_equal(a[key],b[key])
    np.testing.assert_array_equal(a['correct'],[1,1,0])
    out=BASE/'r1_batches_v1_preflight';out.mkdir(exist_ok=False)
    result={'status':'all_real_task_batch_indices_masks_labels_targets_and_synthetic_loss_oracles_verified',
        'counts':counts,'sources_sha256':{str(p):sha(p) for p in [Path(__file__),Path(__file__).with_name('r1_model_v1.py'),
            Path(__file__).with_name('verify_r1_case_v1.py')]},
        'scope':'Fulldata geometry/label alignment plus syntheticlogit arithmetic; no actualmodel forward, taskquality or stochasticdraw.'}
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':main()
