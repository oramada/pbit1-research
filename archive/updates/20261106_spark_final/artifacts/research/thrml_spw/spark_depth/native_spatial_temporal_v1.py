"""Real-head K1 versus actual finite-field K4 word at equal observations."""
import os
os.environ['JAX_PLATFORMS']='cpu'
if hasattr(os,'sched_setaffinity'):os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[-2:])
from pathlib import Path
import json,hashlib,time
import numpy as np
from .prepare_v1 import OUT,sha
from research.thrml_spw.native import NativeProbabilityBank

def main():
    root=OUT/'native_spatial_temporal_v1';root.mkdir(exist_ok=False);begin=time.monotonic()
    protocol={'source_sha256':sha(__file__),'native_source_sha256':sha(Path('research/thrml_spw/native.py')),
        'reference_definition_sha256':sha(Path('research/thrml_spw/broad/parameter_program_v1.py')),'input_sha256':sha(OUT/'native_head_v1/inputs.npz'),
        'seeds':list(range(2026111201,2026111206)),'K1_samples':32768,'K4_samples':8192,
        'K4_definition':'Quantize normalized finite-programprobabilityto4bitword. Fourindependentendpoint-biasedspins withpositionalcoefficients1,2,4,8. Noanalogprobabilityreplicas.',
        'scope':'OfficialTHRML, real8x768headweightbank,8fixedhiddenvectors,5seeds. Equal201326592deviceobservationspermethodperseed. K1uses12bitcontrol/scalar;K4uses4bitword/scalar plussharedendpointfields. Countdifferentcontrolprecision. No fullmodelquality orhardwareenergyclaim.'}
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    with np.load(OUT/'native_head_v1/inputs.npz',allow_pickle=False)as f:d={k:f[k].copy()for k in f.files}
    X=d['hidden'].astype(np.float64);W=d['original_weights'].astype(np.float64);low=1/(1+np.exp(8.));coeff=(2**np.arange(4)).astype(np.int64);records=[];controls=[]
    for layout in ['storage_flat','transposed_matrix']:
        field=d[layout+'_codes'].astype(np.float32)*np.float32(8/4095)-np.float32(4);p=1/(1+np.exp(-2*field.astype(np.float64)));lo=d[layout+'_lower'].astype(np.float64);span=d[layout+'_span'].astype(np.float64)
        words=np.rint(np.clip((p-low)/(1-2*low),0,1)*15).astype(np.uint8)
        bitplanes=((words[None]>>np.arange(4)[:,None,None])&1).astype(bool);p4=np.where(bitplanes,1-low,low)
        banks={'K1':NativeProbabilityBank(p),'K4':NativeProbabilityBank(p4)}
        means={'K1':p,'K4':low+(1-2*low)*words/15.};var_coeff={'K1':p*(1-p),'K4':np.full(W.shape,85/225*low*(1-low))}
        exact_int4=lo+span*words/15.
        controls.append({'layout':layout,'INT4_exact_word_MSE':float(np.mean((X@(exact_int4-W).T)**2))})
        for method,bank in banks.items():
            T=protocol[method+'_samples'];assert bank.logical_pbits==(1 if method=='K1'else 4)*6144
            assert bank.logical_pbits*T==201326592
            theoretical_bias=X@(lo+span*means[method]-W).T
            predicted=float(np.mean(theoretical_bias**2)+(X**2@(span**2*var_coeff[method]/T).T).mean())
            for seed in protocol['seeds']:
                states=bank.sample(seed,T,1)[0];counts=states.sum(axis=0,dtype=np.int32)
                if method=='K1':q=counts.astype(np.float64)/T
                else:q=np.sum(counts.astype(np.int64)*coeff[:,None,None],axis=0).astype(np.float64)/(15*T)
                estimate=lo+span*q;observed=float(np.mean((X@(estimate-W).T)**2))
                file=root/f'{layout}_{method}_seed{seed}.npz'
                with file.open('xb')as f:np.savez_compressed(f,counts=counts,estimate=estimate)
                records.append({'layout':layout,'method':method,'seed':seed,'pbits':bank.logical_pbits,'samples':T,'observations':bank.logical_pbits*T,
                    'output_MSE_about_original':observed,'predicted_output_MSE_about_original':predicted,'squared_bias':float(np.mean(theoretical_bias**2)),
                    'counts_file_sha256':sha(file),'trajectory_sha256':hashlib.sha256(np.ascontiguousarray(states).tobytes()).hexdigest()})
                del states
                (root/'progress.json').write_text(json.dumps({'layout':layout,'method':method,'seed':seed,'elapsed_seconds':time.monotonic()-begin})+'\n')
                print(json.dumps({'layout':layout,'method':method,'seed':seed,'MSE':observed}),flush=True)
    summary={}
    for layout in ['storage_flat','transposed_matrix']:
        summary[layout]={}
        for method in ['K1','K4']:
            r=[r for r in records if r['layout']==layout and r['method']==method];summary[layout][method]={'mean_output_MSE':float(np.mean([x['output_MSE_about_original']for x in r])),'theoretical_MSE':r[0]['predicted_output_MSE_about_original'],'pbits':r[0]['pbits'],'samples':r[0]['samples']}
    result={'status':'completed_native_equal_observation_K1_K4_realhead_comparison','protocol_sha256':sha(root/'protocol.json'),'records':records,'controls':controls,'summary':summary,'elapsed_seconds':time.monotonic()-begin,'scope':protocol['scope']}
    (root/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
