"""Exact conditional moments for illustrative nonideal versions of real head banks."""
import json,math
import numpy as np
from .prepare_v1 import OUT,sha

def sigmoid(x):return np.exp(-np.logaddexp(0,-x))

def temporal_factor(T,rho):
    if rho==0:return 1.
    s1=rho*(1-rho**(T-1))/(1-rho);s2=rho*(1-T*rho**(T-1)+(T-1)*rho**T)/(1-rho)**2
    return 1+2*s1-2*s2/T

def shared_variance(q,span,X):
    out=np.zeros((len(X),len(q)))
    for head in range(len(q)):
        order=np.argsort(q[head]);p=q[head,order];a=X[:,order]*span[head,order]
        aq=a*p;prefix=np.cumsum(aq,axis=1)-aq
        out[:,head]=np.maximum(0,np.sum(a*a*p+2*a*prefix,axis=1)-np.sum(aq,axis=1)**2)
    return out

def moment(X,lo,span,q,W,T,rho,kappa):
    error=X@(lo+span*q-W).T;independent=(X*X)@(span*span*q*(1-q)).T
    var=((1-kappa)*independent+kappa*shared_variance(q,span,X))*temporal_factor(T,rho)/T
    return {'mean_squared_bias':float(np.mean(error**2)),'mean_estimator_variance':float(var.mean()),'expected_output_MSE':float(np.mean(error**2+var))}

def calibrated_logits(target,gain,offset,probes,rng):
    x=np.array([-2.,0.,2.]);q=sigmoid(gain[...,None]*x+offset[...,None]);counts=rng.binomial(probes,q)
    p=(counts+.5)/(probes+1);y=np.log(p)-np.log1p(-p);ahat=np.clip(np.sum(y*x,axis=-1)/8,.05,20.);bhat=y.mean(-1)
    desired=(target-bhat)/ahat;field=desired/2;codes=np.rint(np.clip((field+4)*4095/8,0,4095)).astype(np.uint16)
    actual=codes.astype(np.float32)*np.float32(8/4095)-np.float32(4)
    return 2*actual.astype(np.float64),float(np.mean((field < -4)|(field > 4)))

def main():
    root=OUT/'nonideal_head_v1';root.mkdir(exist_ok=False)
    protocol={'source_sha256':sha(__file__),'native_inputs_sha256':sha(OUT/'native_head_v1/inputs.npz'),
        'device_seeds':list(range(2026111101,2026111111)),'samples':[128,512,8192,32768,131072],
        'assumptions':'Illustrative gainSD.02,offsetSD.02 inlogitunits; driftSD.002 per sqrt(ageunit),frozen duringmeasurement. Common-uniformmixturekappa=.1; whole-bankholdprobabilityrho=.75 forstationarytemporalcorrelation. Notmeasureddeviceparameters.',
        'calibration':'Threeprobe logits[-2,0,2],256or4096shotsperprobeperdevice; empiricallogitfit withJeffreys smoothing; inversecontrolquantizedbackto12bit[-4,4]. Temporarygain/offsetestimates64bits/device, canbediscardedafterprogramming.',
        'scope':'Exactconditionalmoments forreal6144weightoperator plusbinomialcalibrationprobe simulation. No fullnetworkquality, no physicaltimeunits, no nativeTHRMLclaimforthese nonidealmodels.'}
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    # Independent small covariance and finite-sum checks of the moment formulas.
    p=np.linspace(.05,.95,12).reshape(1,12);span=np.linspace(.2,1,12).reshape(1,12);X=np.cos(np.arange(36)).reshape(3,12)
    K=np.minimum(p[0,:,None],p[0,None,:])-np.outer(p[0],p[0]);coeff=X*span
    np.testing.assert_allclose(shared_variance(p,span,X)[:,0],np.einsum('bi,ij,bj->b',coeff,K,coeff),atol=1e-12)
    np.testing.assert_allclose(temporal_factor(32,.75),1+2*sum((1-k/32)*.75**k for k in range(1,32)),atol=1e-12)
    with np.load(OUT/'native_head_v1/inputs.npz',allow_pickle=False)as d:data={k:d[k].copy()for k in d.files}
    X=data['hidden'].astype(np.float64);W=data['original_weights'].astype(np.float64);records=[]
    for seed in protocol['device_seeds']:
        # Same physical parameter errors paired between layouts.
        rng=np.random.default_rng(seed);gain=rng.normal(1,.02,W.shape);offset=rng.normal(0,.02,W.shape);drift=rng.normal(0,.002*np.sqrt(256),W.shape)
        for layout in ['storage_flat','transposed_matrix']:
            codes=data[layout+'_codes'];target=2*(codes.astype(np.float32)*np.float32(8/4095)-np.float32(4)).astype(np.float64)
            lo=data[layout+'_lower'].astype(np.float64);sp=data[layout+'_span'].astype(np.float64)
            cases=[('ideal',sigmoid(target),0.,0.,0,0.),('temporal_rho075',sigmoid(target),.75,0.,0,0.),
                ('cross_kappa01',sigmoid(target),0.,.1,0,0.),('mismatch',sigmoid(gain*target+offset),0.,0.,0,0.),
                ('mismatch_drift_age256',sigmoid(gain*target+offset+drift),0.,0.,0,0.)]
            for probes in [256,4096]:
                probe_rng=np.random.default_rng(np.random.SeedSequence([seed,probes]))
                adjusted,clamped=calibrated_logits(target,gain,offset,probes,probe_rng)
                cases.extend([(f'calibrated_{probes}',sigmoid(gain*adjusted+offset),0.,0.,3*probes*W.size,clamped),
                    (f'calibrated_{probes}_drift_age256',sigmoid(gain*adjusted+offset+drift),0.,0.,3*probes*W.size,clamped)])
            for name,q,rho,kappa,probe_count,clamped in cases:
                for T in protocol['samples']:
                    records.append({'device_seed':seed,'layout':layout,'scenario':name,'samples':T,
                        **moment(X,lo,sp,q,W,T,rho,kappa),'calibration_probes':probe_count,'requested_control_clamped_fraction':clamped})
    result={'status':'completed_exact_nonideal_head_moments_and_probe_calibration','protocol_sha256':sha(root/'protocol.json'),'records':records,
        'formula_fixtures':'Shared-uniformcovariance checkedagainstexplicit12x12kernel; finiteMarkovfactoragainst32termsum.','scope':protocol['scope']}
    (root/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'records':len(records),'device_seeds':10,'calibration_probes_4096':3*4096*6144}))

if __name__=='__main__':main()
