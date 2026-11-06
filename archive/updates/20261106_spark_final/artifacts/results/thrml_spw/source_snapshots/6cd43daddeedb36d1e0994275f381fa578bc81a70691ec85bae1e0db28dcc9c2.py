"""Independent formulas and checkpoint reads for LFM3 grouping calibration."""
import json,math,time
from collections import Counter
import numpy as np
from safetensors import safe_open
from .prepare_v1 import OUT,sha
from .lfm3_full_preflight_v1 import ROOT

def main():
    root=OUT/'lfm3_calibration_v2';r=json.loads((root/'results.json').read_text());p=json.loads((root/'protocol.json').read_text())
    assert r['protocol_sha256']==sha(root/'protocol.json') and r['activation_moments_sha256']==sha(root/'activation_moments.npz')
    assert p['rows']==list(range(8)) and p['input_length']==129
    assert len(r['policies'])==286 and sum(x['parameters']for x in r['policies'])==1728801536
    assert len({x['name']for x in r['policies']})==286
    checked=[];begin=time.monotonic()
    with np.load(root/'activation_moments.npz')as moments,safe_open(str(ROOT/'model.safetensors'),framework='pt',device='cpu')as weights:
        for policy in r['policies']:
            name=policy['name'];role=policy['selected_role'];kind=policy['kind'];shape=policy['shape'];count=8*129*(16 if '.q_norm.' in name else 4 if '.k_norm.' in name else 1)
            key=role+('.lookup_counts'if kind=='embedding'else'.input_sumsq')
            moment=moments[key].reshape(-1)/count
            assert np.isfinite(moment).all() and (moment>=0).all()
            w=weights.get_tensor(name).float().numpy();assert list(w.shape)==shape and w.size==policy['parameters']
            for variant in policy['variants']:
                g=variant['group'];a=w.T.copy()if variant['layout']=='transposed_matrix'else w;flat=a.reshape(-1);total=0.;chunk=262144//g*g
                for start in range(0,flat.size,chunk):
                    q=flat[start:start+chunk];padded=np.full(math.ceil(q.size/g)*g,np.nan,np.float32);padded[:q.size]=q;blocks=padded.reshape(-1,g)
                    low=np.nanmin(blocks,axis=1);span=np.nanmax(blocks,axis=1)-low
                    lo=np.repeat(low,g)[:q.size];sc=np.repeat(span,g)[:q.size]
                    prob=np.divide(q.astype(np.float64)-lo,sc,out=np.full(q.size,.5,np.float64),where=sc!=0)
                    prob=np.clip(prob,1/(1+np.exp(8)),1-1/(1+np.exp(8)))
                    h=(np.log(prob)-np.log1p(-prob))/2
                    codes=np.rint((np.clip(h,-4,4)+4)*(4095/8)).astype(np.uint16)
                    fields=codes.astype(np.float32)*np.float32(8/4095)-np.float32(4);pp=1/(1+np.exp(-2*fields.astype(np.float64)))
                    variance=sc.astype(np.float64)**2*pp*(1-pp);indices=np.arange(start,start+q.size)
                    if kind=='linear':ii=indices%shape[1]if variant['layout']=='storage_flat'else indices//shape[0]
                    elif kind=='embedding':ii=indices//shape[1]if variant['layout']=='storage_flat'else indices%shape[0]
                    else:ii=indices
                    total+=float(variance@moment[ii])
                assert math.isclose(total,variant['variance_coefficient'],rel_tol=3e-6,abs_tol=1e-12),(name,total,variant)
                checked.append({'name':name,'layout':variant['layout'],'group':g,'relative_error':abs(total-variant['variance_coefficient'])/max(total,1e-30)})
            best=min(policy['variants'],key=lambda v:(v['variance_coefficient'],v['range_groups']))
            assert (best['layout'],best['group'])==(policy['selected_layout'],policy['selected_group'])
            print(name,flush=True)
    output={'status':'all_286_parameter_policies_independently_recomputed','result_sha256':sha(root/'results.json'),'source_sha256':sha(__file__),'checked_variants':len(checked),'max_relative_error':max(x['relative_error']for x in checked),'parameters':1728801536,'range_groups':sum(math.ceil(x['parameters']/x['selected_group'])for x in r['policies']),'choices':{str(k):v for k,v in Counter((x['kind'],x['selected_layout'],x['selected_group'])for x in r['policies']).items()},'elapsed_seconds':time.monotonic()-begin}
    (OUT/'lfm3_calibration_v3_audit.json').write_text(json.dumps(output,indent=2)+'\n');print(json.dumps(output),flush=True)
if __name__=='__main__':main()
