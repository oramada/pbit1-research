"""Recompute every quarter-confirmation bootstrap draw using a second aggregation."""
import json,csv
from pathlib import Path
import numpy as np
from .verify_transformer_control import sha


def main():
    out=Path('results/thrml_spw/quarter_confirmation');p=json.loads((out/'protocol.json').read_text());tokens=np.load(p['input_data_path'])['tokens'];base=np.load(out/'baseline_tokens.npz');data={}
    for c in p['conditions']:
        if c=='native_finite8':
            raw=[np.load(out/'runs'/f'confirmation_seed{s}'/'prefix_tokens.npz') for s in range(5)]
            data[c]=(np.stack([r['prefix12_nll'].mean(-1,dtype=np.float64) for r in raw]),np.stack([(r['prefix12_predictions']==tokens[:,1:]).mean(-1) for r in raw]))
        else:data[c]=(np.repeat(base[c+'_nll'].mean(-1,dtype=np.float64)[None],5,0),np.repeat((base[c+'_predictions']==tokens[:,1:]).mean(-1)[None],5,0))
    with (out/'paired_bootstrap.csv').open() as f:rows=list(csv.DictReader(f))
    assert len(rows)==7;draws=np.load(out/'paired_bootstrap_draws.npz');rng=np.random.default_rng(20261108)
    seed_index=rng.integers(0,5,size=(10000,5));bank_index=rng.integers(0,16,size=(10000,16));count=0
    for row in rows:
        assert int(row['seed'])==20261108 and int(row['resamples'])==10000
        left,right=row['left'],row['right'];prefix=left+'_minus_'+right
        for i,suffix in enumerate(['CE','accuracy']):
            difference=(data[left][i]-data[right][i]).reshape(5,16,8).mean(-1)
            averaged_by_bank=np.take(difference,seed_index,axis=0).mean(1)
            resampled=np.take_along_axis(averaged_by_bank,bank_index,axis=1).mean(1)
            np.testing.assert_allclose(resampled,draws[prefix+'_'+suffix],rtol=1e-12,atol=1e-15)
            lo,hi=np.quantile(resampled,[.025,.975])
            if i==0:
                for value,key in [(difference.mean(),'mean_cross_entropy_difference'),(lo,'bootstrap_CE_lower'),(hi,'bootstrap_CE_upper'),(np.exp(difference.mean()),'relative_perplexity'),(np.exp(lo),'bootstrap_relative_lower'),(np.exp(hi),'bootstrap_relative_upper')]:np.testing.assert_allclose(value,float(row[key]),rtol=1e-12)
            else:
                for value,key in [(100*difference.mean(),'accuracy_difference_pp'),(100*lo,'bootstrap_accuracy_lower_pp'),(100*hi,'bootstrap_accuracy_upper_pp')]:np.testing.assert_allclose(value,float(row[key]),atol=1e-12)
            count+=len(resampled)
    result={'passed':True,'paired_comparisons':7,'bootstrap_draw_values_independently_recovered':count,'source_sha256':sha(__file__),
        'scope':'All paired CE/accuracy draw values and intervals reconstructed from raw token arrays, not validation of population-coverage assumptions'}
    (out/'statistics_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
