"""Direct stochastic head accumulation versus reconstructed weights, same states."""
import json
import numpy as np
from .prepare_v1 import OUT,sha

def main():
    source=OUT/'native_head_v1';dest=OUT/'native_linear_identity_v1';dest.mkdir(exist_ok=False)
    protocol=json.loads((source/'protocol.json').read_text());records=[];outputs={}
    with np.load(source/'inputs.npz',allow_pickle=False)as f:data={k:f[k].copy()for k in f.files}
    X=data['hidden'].astype(np.float64);T=8192
    for layout in protocol['layouts']:
        lo=data[layout+'_lower'].astype(np.float64);span=data[layout+'_span'].astype(np.float64)
        for seed in protocol['seeds']:
            path=source/f'{layout}_seed{seed}_states.npz'
            with np.load(path,allow_pickle=False)as f:states=np.unpackbits(f['packed_states'],axis=0,count=T,bitorder='little')
            assert states.shape==(8192,8,768)
            reconstructed=X@(lo+span*np.sum(states,axis=0,dtype=np.int64)/T).T
            total=np.zeros((8,8),np.float64);squares=np.zeros_like(total)
            for start in range(0,T,128):
                sample_weights=lo[None]+span[None]*states[start:start+128]
                per_sample=(sample_weights.reshape(-1,768)@X.T).reshape(-1,8,8).transpose(0,2,1)
                total+=per_sample.sum(axis=0);squares+=(per_sample*per_sample).sum(axis=0)
            direct=total/T;error=np.abs(direct-reconstructed);assert np.max(error)<1e-9
            se=np.sqrt(np.maximum(0,squares-total*total/T)/(T*(T-1)))
            key=f'{layout}_seed{seed}';outputs[key+'_direct']=direct;outputs[key+'_reconstructed']=reconstructed;outputs[key+'_standard_error']=se
            records.append({'layout':layout,'seed':seed,'samples':T,'pbits':6144,'native_states_sha256':sha(path),'maximum_absolute_R1_R2_difference':float(error.max()),'mean_standard_error':float(se.mean())})
    with(dest/'outputs.npz').open('xb')as f:np.savez_compressed(f,**outputs)
    result={'status':'all10_real_head_native_trajectories_match_direct_and_reconstructed_accumulation','source_sha256':sha(__file__),'inputs_sha256':sha(source/'inputs.npz'),'records':records,'outputs_sha256':sha(dest/'outputs.npz'),
        'scope':'Linearoperatoronly, samefrozen nativeTHRMLsamples and FP64CPUarithmetic. Bylinearity, Xmean(W_t)=mean(XW_t); no variance reduction magicallyarisesfrom changingaccumulationorder. Directexecutioncanavoidmaterializingperweightaverages. No whole-networkR2,latency,energyorphysicaldeviceclaim.'}
    (dest/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'status':result['status'],'max_difference':max(r['maximum_absolute_R1_R2_difference']for r in records)}))

if __name__=='__main__':main()
