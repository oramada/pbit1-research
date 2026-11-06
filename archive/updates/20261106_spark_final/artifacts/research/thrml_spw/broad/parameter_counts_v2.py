"""Same official sampler, exact integer combination of positional counts.

The v1 raw state/count path is unchanged. Integer combination avoids differing
FP64 bitplane summation orders before the explicitly defined reconstruction.
"""
import numpy as np
from .parameter_counts_v1 import ParameterCounts as _ParameterCounts
from .parameter_program_v1 import decode_probability


def reconstruct(program,counts,samples):
    counts=np.asarray(counts)
    if counts.dtype!=np.int32 or counts.ndim!=2 or counts.shape[0] not in (1,4,8):
        raise ValueError('Oneint32countvectorper declaredbitplane required')
    if type(samples)is not int or not 1<=samples<2**31 or np.any((counts<0)|(counts>samples)):
        raise ValueError('Countsand samplecount outside declaredbounds')
    bits=counts.shape[0]
    numerator=np.sum(counts.astype(np.int64)*(1<<np.arange(bits,dtype=np.int64))[:,None],axis=0,dtype=np.int64)
    probability=numerator.astype(np.float64)/(samples*((1<<bits)-1))
    return decode_probability(program,probability)


class ParameterCounts(_ParameterCounts):
    def draw(self,program,bits,samples,seed,parameter_index,replay=False):
        result=super().draw(program,bits,samples,seed,parameter_index,replay)
        result['decoded']=reconstruct(program,result['counts'],samples)
        result['readout']='exactint64positionalnumerator_then_FP64divide_mul_add'
        return result
