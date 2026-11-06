"""Validation-only preparation; no classifier sampling or test scoring."""
from pathlib import Path
import hashlib,json
import numpy as np
OUT=Path('results/thrml_spw/boolean_classifier_preparation')
DATA=Path('data/thrml_spw/positive_mnist/evaluation_data.npz')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def finite_field(p):
    p=np.asarray(p,dtype=np.float64)
    if not np.isfinite(p).all() or np.any((p<0)|(p>1)):raise ValueError('probabilities must lie in [0,1]')
    eps=1/(1+np.exp(8));q=np.clip(p,eps,1-eps)
    f=.5*(np.log(q)-np.log1p(-q));code=np.rint((np.clip(f,-4,4)+4)*255/8).astype(np.uint8)
    field=(-4+8*code.astype(float)/255).astype(np.float32)
    return code,field

def budgets():
    scales=[2,8,32];r1=[49*k for k in scales];r3=[37*k for k in scales]
    assert all(a*2368==b*3136 for a,b in zip(r1,r3))
    per_regime=5*10*1250*sum(r1)*2368
    assert per_regime==304584000000
    # Lossless counter widths: maximum count1568<65536; bank contains eight inputs.
    banks=5*10*3*1250
    payload={'R1_uint16_weight_counts':banks*2368*2,'R3_uint16_AND_and_column_counts':banks*(8*32+8*10+32+10)*2,'R3_uint8_hidden_activation_codes':banks*8*32,'both_F64_final_logits':2*5*10*3*10000*10*8}
    return {'R1_samples':r1,'R3_samples':r3,'R1_active_nodes':2368,'R3_active_nodes':3136,'models':5,'MC_seeds':10,'test_images':10000,'banks_per_case':1250,'bank_size':8,'production_updates_per_regime':per_regime,'production_updates_total':2*per_regime,'complete_independent_replay_updates':2*per_regime,'uncompressed_required_payload_bytes':payload,'uncompressed_required_payload_total':sum(payload.values()),'extra_metadata_and_working_capacity_reserve':250000000,'minimum_free_before_production':sum(payload.values())+250000000}

def main():
    assert not (OUT/'preparation.json').exists();OUT.mkdir(exist_ok=True)
    with np.load(DATA) as data:x=data['validation_inputs'].astype(float)
    assert x.shape==(5000,64) and np.isfinite(x).all()
    lower=float(x.min());span=float(x.max()-lower);assert lower<0 and span>0
    rows=[]
    for seed in range(5):
        source=Path(f'data/thrml_spw/positive_mnist/seed{seed}.npz')
        with np.load(source) as a:
            w0=a['weight_0'].copy();w1=a['weight_1'].copy();b0=a['bias_0'].copy();b1=a['bias_1'].copy()
        assert w0.shape==(64,32) and w1.shape==(32,10)
        c0,f0=finite_field(w0);c1,f1=finite_field(w1)
        h=np.maximum(x@w0.astype(float)+b0,0)
        # A fixed25% safety margin, chosen without test outputs or sampled quality.
        hidden_upper=1.25*float(h.max());assert hidden_upper>0
        target=OUT/f'model{seed}_controls.npz'
        np.savez_compressed(target,weight_0=w0,weight_1=w1,bias_0=b0,bias_1=b1,weight_code_0=c0,weight_code_1=c1,weight_field_0=f0,weight_field_1=f1,input_lower=np.float64(lower),input_span=np.float64(span),hidden_lower=np.float64(0),hidden_span=np.float64(hidden_upper))
        rows.append({'model_seed':seed,'model_path':str(source),'model_sha256':sha(source),'controls_path':str(target),'controls_sha256':sha(target),'validation_hidden_max':float(h.max()),'hidden_upper':hidden_upper})
    result={'stage':'validation_only_preparation_not_sampling_protocol','data_path':str(DATA),'data_sha256':sha(DATA),'source_sha256':sha(__file__),'validation_examples':5000,'input_lower':lower,'input_span':span,'hidden_range_policy':'Per-model [0,1.25*max validation FP64 FP32-weight ReLU output]; fixed before test scoring. Track stochastic clipping separately.','field_policy':'All nodes free, field bound4,256-level uniform field grid, beta1; endpoints approximate, never clamped.','execution':'R1 reconstructs per-bank weight means then F64 affine operators/ReLU; R3 uses integer AND plus weight-column popcounts and F64 affine readout before ReLU; no intermediate hidden F64 arrays required on disk.','resources':budgets(),'models':rows,'production_ready':False,'remaining':['Seal producer, independent public replay, causal path and statistics sources before scoring.','Validate native counter observer and affine readout on bounded synthetic cases.','Verify actual persistent free capacity against minimum, including metadata/working reserve.','Freeze all controls, keys and analysis choices in a separate production protocol.'],'scope':'Preparation uses validation inputs only. Test data were already consumed in prior studies; no fresh-confirmation claim. Full classifier quality remains unmeasured here.'}
    (OUT/'preparation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'prepared':True,'production_ready':False,'resources':result['resources']},indent=2))
if __name__=='__main__':main()
