"""Frozen mean-plus-resampling selection rule for the new quarter-model study."""
import numpy as np
from .confirmation_statistics import bootstrap_paired_difference


def summarize_profile(native_nll,native_pred,reference_nll,reference_pred,tokens,samples,bootstrap_seed=20261111):
    nll=np.asarray(native_nll);pred=np.asarray(native_pred);reference_nll=np.asarray(reference_nll)
    if nll.shape!=(5,128,64) or pred.shape!=nll.shape or reference_nll.shape!=(128,64):raise ValueError('Expected five seeds and128 windows of64 target positions')
    expected=tokens[:,1:];loss=nll.mean(-1,dtype=np.float64);ref=np.broadcast_to(reference_nll.mean(-1,dtype=np.float64),loss.shape)
    correct=(pred==expected[None]).mean(-1);ref_correct=np.broadcast_to((reference_pred==expected).mean(-1),loss.shape)
    delta=bootstrap_paired_difference(loss,ref,seed=bootstrap_seed);accuracy=bootstrap_paired_difference(correct,ref_correct,seed=bootstrap_seed)
    upper=float(np.quantile(delta,.975));lower=float(np.quantile(accuracy,.025))
    ce_difference=float(np.mean(loss-ref));acc_difference=float(np.mean(correct-ref_correct));ce_seed=loss.mean(-1)
    point=bool(ce_difference<=np.log(1.05) and acc_difference>=-.01)
    passing=bool(point and upper<=np.log(1.05) and lower>=-.01)
    record={'samples':int(samples),'cross_entropy_mean':float(loss.mean()),'cross_entropy_mc95_margin':float(1.96*ce_seed.std(ddof=1)/np.sqrt(5)),
        'perplexity':float(np.exp(loss.mean())),'relative_perplexity':float(np.exp(ce_difference)),'token_accuracy_mean':float(correct.mean()),
        'accuracy_difference':acc_difference,'point_tolerance_passed':point,'bootstrap_CE_upper':upper,'bootstrap_relative_upper':float(np.exp(upper)),
        'bootstrap_accuracy_difference_lower':lower,'passes':passing,'bootstrap_seed':bootstrap_seed,'bootstrap_resamples':10000}
    return record,{'CE_difference':delta,'accuracy_difference':accuracy}
