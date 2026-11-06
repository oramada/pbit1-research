"""Offline integer/command/readout audit from preserved public trajectories."""
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    out=Path('results/thrml_spw/lfm3_1p7b/group128_preflight')
    r=json.loads((out/'results.json').read_text());assert r['status']=='all16_grouped_checks_passed'
    assert r['tied_embedding_head_codes_exact'] and len(r['receipts'])==16
    for name,h in r['artifacts_sha256'].items():assert sha(out/name)==h
    entries=0;tiles=0;coordinates=0
    for row in r['receipts']:
        if row['mode']!='native':continue
        call=row['call'];data=np.load(out/f'input_{call}.npz');x=data['x'];w=data['w'];kind=row['kind']
        scale=float(np.max(np.abs(x)))/127 if np.any(x) else 1.0
        quant=np.rint(x/scale).clip(-127,127).astype(np.int64)
        k=min(128,w.shape[0]);groups=w.shape[0]//k;parts=[]
        for tile,first in enumerate(range(0,w.shape[1],64)):
            width=min(64,w.shape[1]-first);wt=w[:,first:first+width].astype(np.float64)
            wt=np.pad(wt,((0,0),(0,64-width)))
            wg=wt.reshape(groups,k,64);lo=np.min(wg,axis=1);span=np.max(wg,axis=1)-lo
            if kind in ['embedding','elementwise']:
                lo=np.pad(data['lower'][first:first+width].astype(np.float64),(0,64-width))[None,:]
                span=np.pad(data['span'][first:first+width].astype(np.float64),(0,64-width))[None,:]
            p=np.divide(wg-lo[:,None,:],span[:,None,:],out=np.full_like(wg,.5),where=span[:,None,:]!=0).clip(1e-12,1-1e-12)
            h=((np.log(p)-np.log1p(-p))/2).clip(-4,4)
            codes=np.rint((h+4)*255/8).clip(0,255).astype(np.uint8)
            raw=np.load(out/'raw_states'/f'call{call}_tile{tile}.npz')
            np.testing.assert_array_equal(codes,raw['codes'])
            states=raw['states'];assert states.dtype==np.bool_ and states.shape==(row['samples'],groups,k,64)
            if kind=='linear':xt=quant;gx=xt.reshape(xt.shape[0],groups,k)
            else:
                xt=np.pad(quant[:,:,first:first+width],((0,0),(0,0),(0,64-width)))
                gx=xt.reshape(xt.shape[0],groups,k,64)
            np.testing.assert_array_equal(xt,raw['inputs'])
            total=np.zeros(raw['total'].shape,dtype=np.int64)
            for group in range(groups):
                count=states[:,group].astype(np.int64).sum(axis=0)
                if kind=='linear':total[:,group]=gx[:,group]@count
                else:total[:,group]=np.sum(gx[:,group]*count[None,:,:],axis=1)
            np.testing.assert_array_equal(total,raw['total'].astype(np.int64))
            y=np.zeros((x.shape[0],64),dtype=np.float64)
            for group in range(groups):
                offset=gx[:,group].sum(1)[:,None] if kind=='linear' else gx[:,group].sum(1)
                y+=(total[:,group].astype(np.float64)/row['samples']*span[group]+offset*lo[group])*scale
            parts.append(y[:,:width]);entries+=total.size;tiles+=1;coordinates+=wt.size
        np.testing.assert_allclose(np.concatenate(parts,axis=1),np.load(out/f'output_{call}.npy'),rtol=2e-12,atol=2e-12)
    audit={'status':'all_saved_public_states_commands_and_readouts_verified','source_sha256':sha(Path(__file__)),
        'results_sha256':sha(out/'results.json'),'native_cases':8,'tile_cases':tiles,
        'integer_accumulator_entries_verified':entries,'programmed_coordinates_including_padding':coordinates,
        'scope':'Offline exact verification using counts of preserved public states and separate integer dot products; finite commands and external readouts checked. No task-quality claim.'}
    (out/'independent_audit.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps(audit))


if __name__=='__main__':main()
