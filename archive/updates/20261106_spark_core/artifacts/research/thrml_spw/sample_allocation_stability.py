"""Post-hoc validation-only schedule stability; never changes the frozen choice."""
import json
from pathlib import Path
import numpy as np
from .sample_allocation_math import allocate
from .transformer_block_gpu import sha,save_json
OUT=Path('results/thrml_spw/sample_allocation')
def main():
    frozen=sha(OUT/'allocation.json');a=json.loads((OUT/'allocation.json').read_text());m=json.loads((OUT/'validation_manifest.json').read_text());ref=np.load(OUT/'validation_baseline.npz')['mean_nll'].astype(float).mean(-1);loss=np.empty((24,5,32))
    for r in m['cases']:loss[r['stage'],r['seed']]=np.load(r['tokens'])['nll'].astype(float).mean(-1)
    rows=[]
    for mode,count in [('seed',5),('bank',4)]:
        for drop in range(count):
            si=np.array([s for s in range(5) if mode!='seed' or s!=drop]);wi=np.array([w for w in range(32) if mode!='bank' or w//8!=drop]);delta=loss[:,si][:,:,wi].mean((1,2))-ref[wi].mean();coeff=np.maximum(delta,1e-4);choice=allocate(coeff,a['weight_units'],288);sample=[256*v for v in choice['levels']];rows.append({'drop_unit':mode,'drop_index':drop,'samples':sample,'layers_different_from_frozen':sum(x!=y for x,y in zip(sample,a['samples'])),'objective_on_reduced_validation':choice['objective'],'uniform_objective_on_reduced_validation':float(coeff.sum()/4)})
    assert sha(OUT/'allocation.json')==frozen;r={'frozen_allocation_sha256':frozen,'native_observations_added':0,'test_scores_read':False,'frozen_allocation_changed':False,'rows':rows,'source_sha256':sha(__file__),'scope':'Post-hoc descriptive leave-one-seed/bank validation sensitivity only. Alternative schedules never evaluated on test, never selected, and do not replace the preregistered frozen schedule. Not a confidence interval or neural-quality result.'};save_json(OUT/'selection_stability.json',r);print(json.dumps({'frozen_unchanged':True,'schedule_changed_layers':[v['layers_different_from_frozen'] for v in rows]},indent=2))
if __name__=='__main__':main()
