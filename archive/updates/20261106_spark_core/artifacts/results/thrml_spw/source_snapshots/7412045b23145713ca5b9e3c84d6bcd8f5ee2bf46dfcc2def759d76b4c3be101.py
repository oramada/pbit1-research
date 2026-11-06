"""Recorded IID binary activity; these are not physical switching-energy laws."""
from pathlib import Path
import hashlib
import numpy as np
OUT=Path('results/thrml_spw/state_activity')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def moments(probability,t,stationary=False):
    p=np.asarray(probability,float);v=p*(1-p);q=2*v
    if stationary:return t*q,t*q*(1-q)+2*(t-1)*(v-4*v*v)
    if t==1:return p,v
    return p+(t-1)*q,v+(t-1)*q*(1-q)+2*v*(1-2*p)+2*(t-2)*(v-4*v*v)
def full_ledger(entries):
    rows=[]
    for entry in entries:
        n=entry['weights'];a=np.load(entry['single_controls']['path']);b=(-4+a['codes'].astype(float)*8/255).astype(np.float32);p=np.exp(-np.logaddexp(0,-2*b.astype(float)));single=(float(p.sum()),float(np.sum(p*(1-p))))
        word=np.load(entry['positional_controls']['path'])['codes'];ones=sum(int(np.count_nonzero(word&(1<<k))) for k in range(4));ends=np.exp(-np.logaddexp(0,-2*np.array([-4.,4.])));P=ones*ends[1]+(4*n-ones)*ends[0];V=ones*ends[1]*(1-ends[1])+(4*n-ones)*ends[0]*(1-ends[0]);multi=(float(P),float(V))
        for budget in [4,16,64,256]:
            for k,(psum,vsum) in [(1,single),(4,multi)]:
                t=budget//k;rows.append({'stage':entry['stage'],'K':k,'samples':t,'weights':n,'logical_free_nodes':k*n,'updates_per_weight':budget,'scheduled_updates_per_bank':budget*n,'expected_ones_in_first_record':psum,'sum_p_one_minus_p':vsum,'expected_reset_changes_per_bank':psum+2*(t-1)*vsum,'expected_retained_changes_per_bank':2*t*vsum,'expected_reset_0_to_1':psum+(t-1)*vsum,'expected_reset_1_to_0':(t-1)*vsum,'expected_retained_0_to_1':t*vsum,'expected_retained_1_to_0':t*vsum})
    totals=[]
    for budget in [4,16,64,256]:
        for k in [1,4]:
            selected=[r for r in rows if r['updates_per_weight']==budget and r['K']==k];n=sum(r['weights'] for r in selected);reset=sum(r['expected_reset_changes_per_bank'] for r in selected);steady=sum(r['expected_retained_changes_per_bank'] for r in selected);totals.append({'updates_per_weight':budget,'K':k,'weights':n,'samples':budget//k,'scheduled_updates_per32_context_forward':4*budget*n,'expected_reset_changes_per32_context_forward':4*reset,'expected_retained_changes_per32_context_forward':4*steady,'reset_changes_per_weight_per_bank':reset/n,'retained_changes_per_weight_per_bank':steady/n})
    return rows,totals
