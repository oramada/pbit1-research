"""Official THRML unary state-count observer with bounded tile workspace."""
from functools import lru_cache
import numpy as np
import jax
import jax.numpy as jnp
from thrml import SamplingSchedule,sample_with_observation,sample_states
from research.thrml_spw.native import NativeProbabilityBank
from research.thrml_spw.controlled_positive_native import set_fields
from research.thrml_spw.lfm3.counted_bank import BankCounts
from .parameter_program_v1 import fields,word_codes,validate,decode_probability,resources


class ParameterCounts:
    def __init__(self,tile_size):
        if type(tile_size)is not int or tile_size<1:raise ValueError('Positive tile width required')
        self.tile_size=tile_size
        self.bank=NativeProbabilityBank(np.full(tile_size,.5))
        assert self.bank.logical_pbits==tile_size and len(self.bank.clamped_indices)==0

    @lru_cache(maxsize=12)
    def functions(self,samples):
        if type(samples)is not int or not 1<=samples<2**31:raise ValueError('int32-safe sample count required')
        bank=self.bank;initial=[jnp.zeros(self.tile_size,dtype=jnp.bool_)]
        schedule=SamplingSchedule(1,samples,1)
        def counts(key,h):
            observer=BankCounts(bank.readout,bank.logical_pbits)
            total,_=sample_with_observation(key,set_fields(bank,h),schedule,initial,[],observer.init(),observer)
            return total
        def public(key,h):
            return sample_states(key,set_fields(bank,h),schedule,initial,[],[bank.readout])[0]
        return jax.jit(counts),jax.jit(public)

    def draw(self,program,bits,samples,seed,parameter_index,replay=False):
        validate(program)
        if bits not in (1,4,8) or not 0<=seed<2**32 or not 0<=parameter_index<2**32:
            raise ValueError('Validformat,MCkeyandcanonicalparameterindexrequired')
        run,public=self.functions(samples)
        n=program['codes'].size
        words=None if bits==1 else word_codes(program,bits)
        counts=np.empty((bits,n),np.int32);raw=[]
        base=jax.random.fold_in(jax.random.key(seed),parameter_index)
        for tile,begin in enumerate(range(0,n,self.tile_size)):
            end=min(begin+self.tile_size,n);width=end-begin
            for bit in range(bits):
                h=(fields(program['codes'][begin:end]) if bits==1 else
                   np.where((words[begin:end]>>bit)&1,4,-4).astype(np.float32))
                h=np.pad(h,(0,self.tile_size-width))
                key=jax.random.fold_in(jax.random.fold_in(base,tile),bit)
                actual=np.asarray(run(key,jnp.asarray(h)))
                assert actual.shape==(self.tile_size,) and actual.dtype==np.int32
                assert np.all((actual>=0)&(actual<=samples))
                counts[bit,begin:end]=actual[:width]
                if replay:
                    states=np.asarray(public(key,jnp.asarray(h)))
                    np.testing.assert_array_equal(actual,states.sum(axis=0,dtype=np.int64))
                    raw.append({'tile':tile,'begin':begin,'width':width,'bit':bit,'fields':h,
                                'states':states,'padded_counts':actual.copy()})
        weights=(1<<np.arange(bits,dtype=np.int64))/((1<<bits)-1)
        p=np.einsum('bn,b->n',counts.astype(np.float64),weights)/samples
        decoded=decode_probability(program,p)
        return {'counts':counts,'decoded':decoded,'raw':raw,
                'resources':resources(program,bits,samples,self.tile_size)}
