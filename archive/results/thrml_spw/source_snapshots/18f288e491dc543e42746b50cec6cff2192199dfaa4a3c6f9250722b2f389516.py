"""The frozen crossed-seed/bank rule for this study's 64 reserved windows."""
import numpy as np
from .confirmation_statistics import bootstrap_paired_difference


def summarize(nll,pred,reference_nll,reference_pred,tokens):
    if nll.shape!=(5,64,64) or pred.shape!=nll.shape or reference_nll.shape!=(64,64):
        raise ValueError('Frozen study requires five seeds and64 windows of64 targets')
    truth=tokens[:,1:];loss=nll.mean(-1,dtype=float);ref=np.broadcast_to(reference_nll.mean(-1,dtype=float),loss.shape)
    correct=(pred==truth[None]).mean(-1);refcorrect=np.broadcast_to((reference_pred==truth).mean(-1),loss.shape)
    draws=bootstrap_paired_difference(loss,ref,seed=2026114701);adraws=bootstrap_paired_difference(correct,refcorrect,seed=2026114701)
    difference=float(np.mean(loss-ref));accuracy=float(np.mean(correct-refcorrect));upper=float(np.quantile(draws,.975));lower=float(np.quantile(adraws,.025))
    point=bool(difference<=np.log(1.05) and accuracy>=-.01)
    record={'samples':1024,'cross_entropy_mean':float(loss.mean()),'cross_entropy_mc95_margin':float(1.96*loss.mean(-1).std(ddof=1)/np.sqrt(5)),
        'perplexity':float(np.exp(loss.mean())),'relative_perplexity':float(np.exp(difference)),'token_accuracy_mean':float(correct.mean()),'accuracy_difference':accuracy,
        'point_tolerance_passed':point,'bootstrap_CE_upper':upper,'bootstrap_relative_upper':float(np.exp(upper)),'bootstrap_accuracy_difference_lower':lower,
        'passes':bool(point and upper<=np.log(1.05) and lower>=-.01),'bootstrap_seed':2026114701,'bootstrap_resamples':10000}
    return record,{'CE_difference':draws,'accuracy_difference':adraws}
