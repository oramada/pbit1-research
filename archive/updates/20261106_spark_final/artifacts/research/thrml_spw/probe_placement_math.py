"""Weight-only nominal Fisher design for two finite symmetric calibration probes."""
import numpy as np


def decode(code):return (-4+np.asarray(code,dtype=float)*8/255).astype(np.float32)


def weighted_moments(codes,group_span):
    b=decode(codes).astype(float);span=np.repeat(np.asarray(group_span,np.float32),128,0)[:b.shape[0]].astype(float)
    p=1/(1+np.exp(-2*b));s=(2*span*p*(1-p))**2
    return {'N':int(b.size),'A0':float(s.sum()),'A1':float((s*b).sum()),'A2':float((s*b*b).sum())}


def score_pair(moments,low,high,samples_each=2048):
    x,y=float(decode(np.array(low,np.uint8))),float(decode(np.array(high,np.uint8)))
    if not x<y or samples_each<1:raise ValueError('Distinct ordered probes and positive count required')
    qx,qy=1/(1+np.exp(-2*x)),1/(1+np.exp(-2*y))
    vx=.25/(samples_each*qx*(1-qx));vy=.25/(samples_each*qy*(1-qy))
    a0,a1,a2=[moments[k] for k in ['A0','A1','A2']]
    return (vx*(y*y*a0-2*y*a1+a2)+vy*(x*x*a0-2*x*a1+a2))/((y-x)**2*moments['N'])


def design(codes,span):
    m=weighted_moments(codes,span)
    candidates=[{'low_code':k,'high_code':255-k,'nominal_first_order_weight_MSE':score_pair(m,k,255-k)} for k in range(128)]
    chosen=min(candidates,key=lambda c:(c['nominal_first_order_weight_MSE'],c['low_code']))
    return {'moments':m,'chosen':chosen,'fixed_baseline':candidates[96],'candidates':candidates,
        'scope':'Nominal unit-gain/zero-offset asymptotic delta/Fisher criterion. Ignores finite estimator bias/guards, bounded inverse control, quantization and actual device variation. Weights/ranges only, no device truth or task scores.'}
