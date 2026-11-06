"""Quantized-probe binomial calibration: MLE versus an explicitly assumed prior."""
import json
import numpy as np
from .prepare_v1 import OUT,sha
from .nonideal_head_v1 import sigmoid,moment

def quantized_logit(z):
    field=np.asarray(z)/2;codes=np.rint(np.clip((field+4)*4095/8,0,4095)).astype(np.uint16)
    return 2*(codes.astype(np.float32)*np.float32(8/4095)-np.float32(4)).astype(np.float64)

def fit(counts,N,x,prior):
    a=np.ones(counts.shape[:-1]);b=np.zeros_like(a);precision=2500. if prior else 0.
    for _ in range(10):
        p=sigmoid(a[...,None]*x+b[...,None]);e=N*p-counts;v=N*p*(1-p)
        ga=np.sum(e*x,axis=-1)+precision*(a-1);gb=np.sum(e,axis=-1)+precision*b
        aa=np.sum(v*x*x,axis=-1)+precision;ab=np.sum(v*x,axis=-1);bb=np.sum(v,axis=-1)+precision;det=aa*bb-ab*ab
        a-= (bb*ga-ab*gb)/det;b-= (aa*gb-ab*ga)/det
    return a,b

def main():
    root=OUT/'calibration_prior_v1';root.mkdir(exist_ok=False);protocol={'source_sha256':sha(__file__),'moment_source_sha256':sha(__import__('pathlib').Path(__file__).with_name('nonideal_head_v1.py')),
        'inputs_sha256':sha(OUT/'native_head_v1/inputs.npz'),'device_seeds':list(range(2026111101,2026111111)),
        'samples':32768,'probe_counts':[256,4096,65536],'gain_SD':.02,'offset_SD':.02,
        'methods':['uncalibrated','binomial_MLE','binomial_MAP_assumed_population_prior'],
        'scope':'Illustrative independentdeviceerrors. MAP assumescorrectN(1,.02^2)gain,N(0,.02^2)offsetpopulationprior, notlearnedfromhardware. Probe logits use12bitDACquantization. ExactoperatorconditionalMSE, notneuralquality. Retainprobeobservations; countlogicalshotsandtemporaryestimatorstate.'}
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    with np.load(OUT/'native_head_v1/inputs.npz')as f:data={k:f[k].copy()for k in f.files}
    X=data['hidden'].astype(np.float64);W=data['original_weights'].astype(np.float64);x=quantized_logit(np.array([-2.,0.,2.]));results=[];raw={}
    for seed in protocol['device_seeds']:
        rng=np.random.default_rng(seed);gain=rng.normal(1,.02,W.shape);offset=rng.normal(0,.02,W.shape)
        for layout in ['storage_flat','transposed_matrix']:
            z=2*(data[layout+'_codes'].astype(np.float32)*np.float32(8/4095)-np.float32(4)).astype(np.float64);lo=data[layout+'_lower'].astype(np.float64);sp=data[layout+'_span'].astype(np.float64)
            baseline=moment(X,lo,sp,sigmoid(gain*z+offset),W,32768,0,0)
            results.append({'seed':seed,'layout':layout,'method':'uncalibrated','probes_per_operating_point':0,'calibration_shots':0,**baseline})
            for N in protocol['probe_counts']:
                probe_rng=np.random.default_rng(np.random.SeedSequence([seed,N,1]));counts=probe_rng.binomial(N,sigmoid(gain[...,None]*x+offset[...,None]));raw[f'{layout}_{seed}_{N}']=counts.astype(np.uint32)
                for prior in [False,True]:
                    a,b=fit(counts,N,x,prior);requested=(z-b)/a;adjusted=quantized_logit(requested);physical=sigmoid(gain*adjusted+offset)
                    results.append({'seed':seed,'layout':layout,'method':'MAP'if prior else'MLE','probes_per_operating_point':N,'calibration_shots':3*N*W.size,
                        'calibration_shots_over_one_inference_sampling_budget':3*N/32768,'temporary_gain_offset_estimate_bits':64*W.size,'persistent_corrected_control_bits':12*W.size,
                        'clamped_fraction':float(np.mean((requested < -8)|(requested>8))),**moment(X,lo,sp,physical,W,32768,0,0)})
    with(root/'probe_counts.npz').open('xb')as f:np.savez_compressed(f,probe_logit=x,**raw)
    result={'status':'completed_quantized_probe_MLE_MAP_comparison','protocol_sha256':sha(root/'protocol.json'),'probe_counts_sha256':sha(root/'probe_counts.npz'),'records':results,'scope':protocol['scope']}
    (root/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    for layout in ['storage_flat','transposed_matrix']:
        print(layout)
        for method,N in [('uncalibrated',0)]+[(m,n)for n in protocol['probe_counts']for m in ['MLE','MAP']]:
            r=[r['expected_output_MSE']for r in results if r['layout']==layout and r['method']==method and r['probes_per_operating_point']==N];print(method,N,float(np.mean(r)))

if __name__=='__main__':main()
