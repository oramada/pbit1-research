"""Independently recover every saved crossed-bootstrap draw from raw token loss."""
import csv,json,hashlib
from pathlib import Path
import numpy as np


def main():
    out=Path('results/thrml_spw/transformer_confirmation');p=json.loads((out/'protocol.json').read_text());tokens=np.load(p['input_data_path'])['tokens']
    base=np.load(out/'baseline_tokens.npz');data={}
    for c in ('FP32','affine_zero_block128_4'):
        data[c]=(np.repeat(base[c+'_nll'].mean(-1,dtype=np.float64)[None],5,0),np.repeat((base[c+'_predictions']==tokens[:,1:]).mean(-1)[None],5,0))
    for c in ('native_float','native_finite8'):
        raw=[np.load(out/'runs'/f'{c}_seed{s}'/'prefix_tokens.npz') for s in range(5)]
        data[c]=(np.stack([r['prefix4_nll'].mean(-1,dtype=np.float64) for r in raw]),np.stack([(r['prefix4_predictions']==tokens[:,1:]).mean(-1) for r in raw]))
    with (out/'paired_bootstrap.csv').open() as f:rows=list(csv.DictReader(f))
    assert len(rows)==5;saved=np.load(out/'paired_bootstrap_draws.npz')
    rng=np.random.default_rng(20261106);si=rng.integers(0,5,size=(10000,5));bi=rng.integers(0,16,size=(10000,16));count=0
    for r in rows:
        left,right=r['left'],r['right'];prefix=left+'_minus_'+right
        for i,suffix in enumerate(('CE','accuracy')):
            difference=(data[left][i]-data[right][i]).reshape(5,16,8).mean(-1)
            seed_average=np.take(difference,si,axis=0).mean(axis=1)
            values=np.take_along_axis(seed_average,bi,axis=1).mean(axis=1)
            np.testing.assert_allclose(values,saved[prefix+'_'+suffix],rtol=1e-12,atol=1e-15)
            lo,hi=np.percentile(values,[2.5,97.5])
            if i==0:
                for actual,k in [(difference.mean(),'mean_cross_entropy_difference'),(lo,'bootstrap_CE_lower'),(hi,'bootstrap_CE_upper'),(np.exp(lo),'bootstrap_relative_lower'),(np.exp(hi),'bootstrap_relative_upper')]:np.testing.assert_allclose(actual,float(r[k]),rtol=1e-12)
            else:
                for actual,k in [(100*difference.mean(),'mean_accuracy_difference_pp'),(100*lo,'bootstrap_accuracy_lower_pp'),(100*hi,'bootstrap_accuracy_upper_pp')]:np.testing.assert_allclose(actual,float(r[k]),atol=1e-12)
            count+=len(values)
    result={'passed':True,'paired_comparisons':5,'bootstrap_draw_values_recovered':count,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scope':'Every saved CE/accuracy resample and interval independently recomputed from raw token arrays; validates arithmetic, not population coverage assumptions'}
    (out/'statistics_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
