"""Explicit counterfactual controller fields and telescoping weight-error moments."""
import numpy as np
METHODS=('ideal8','measured8','oracle8','oracle_continuous_bound4')


def fields(nominal,gain,offset,measured_codes):
    nominal=np.asarray(nominal,np.float32);gain=np.asarray(gain,float);offset=np.asarray(offset,float)
    if nominal.shape!=gain.shape or gain.shape!=offset.shape or gain.shape!=np.shape(measured_codes):raise ValueError('One control/device value per coordinate required')
    if np.any(gain<=0) or not all(np.isfinite(x).all() for x in [nominal,gain,offset]):raise ValueError('Finite fields and positive gains required')
    inverse=(nominal.astype(float)-offset)/gain
    code=np.rint((np.clip(inverse,-4,4)+4)*255/8).astype(np.uint8)
    decode=lambda c:(-4+np.asarray(c,dtype=float)*8/255).astype(np.float32)
    out={'ideal8':nominal,'measured8':(gain*decode(measured_codes).astype(float)+offset).astype(np.float32),
         'oracle8':(gain*decode(code).astype(float)+offset).astype(np.float32),
         'oracle_continuous_bound4':(gain*np.clip(inverse,-4,4)+offset).astype(np.float32)}
    return out,code,inverse


def moments(probabilities,span):
    # This exact telescoping identity retains signed cross terms. It is not an
    # additive independent-noise attribution or an additive task-loss decomposition.
    q=probabilities;span=np.asarray(span,float)
    estimate=span*(q['measured8']-q['oracle8'])
    grid=span*(q['oracle8']-q['oracle_continuous_bound4'])
    bound=span*(q['oracle_continuous_bound4']-q['ideal8'])
    total=span*(q['measured8']-q['ideal8'])
    np.testing.assert_allclose(total,estimate+grid+bound,atol=1e-14,rtol=1e-12)
    terms={'measured_minus_known_SSE':float(np.sum(estimate**2)),'oracle_grid_SSE':float(np.sum(grid**2)),'oracle_range_SSE':float(np.sum(bound**2)),
        'twice_estimate_grid':float(2*np.sum(estimate*grid)),'twice_estimate_range':float(2*np.sum(estimate*bound)),'twice_grid_range':float(2*np.sum(grid*bound))}
    mse=float(np.sum(total**2));closure=float(sum(terms.values())-mse)
    return {'weights':int(total.size),'total_SSE':mse,**terms,'SSE_closure_error':closure}
