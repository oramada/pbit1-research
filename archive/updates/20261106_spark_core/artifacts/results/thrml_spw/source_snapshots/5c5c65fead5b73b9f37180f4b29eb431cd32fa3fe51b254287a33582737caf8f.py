"""Piecewise native linear-output accumulation before network nonlinearities.

Each segment is an actual public THRML sampling call at fixed fields. Independent
unary spins mix in one sweep, so restart states do not discard a burn-in cost or
model temporal persistence. Correlated hardware needs a different transition model.
"""
from functools import lru_cache
import jax
import jax.numpy as jnp
from .controlled_positive_native import ControlledPositiveNetwork


class ClockedPositiveNetwork(ControlledPositiveNetwork):
    @lru_cache(maxsize=4)
    def dataset_segments(self,segment_samples):
        first,last=[layer.with_fields(segment_samples) for layer in self.layers]
        split=self.free_split
        def run(base_key,input_batches,fields0,fields1):
            # [batch, segment, local-node] fields follow a serial absolute clock.
            if fields0.ndim!=3 or fields1.ndim!=3 or fields0.shape[:2]!=fields1.shape[:2]:
                raise ValueError('Matching batch and segment field schedules required')
            if fields0.shape[-1]!=split or fields1.shape[-1]!=len(self.layers[1].bank.free_indices):
                raise ValueError('One field per layer free node required')
            batches=jnp.arange(input_batches.shape[0]);segments=jnp.arange(fields0.shape[1])
            def one(batch,x,f0,f1):
                kb=jax.random.fold_in(base_key,batch)
                def layer_mean(fn,layer,inputs,fields):
                    kl=jax.random.fold_in(kb,layer)
                    keys=jax.vmap(lambda s:jax.random.fold_in(kl,s))(segments)
                    # Each fn includes the deterministic neural bias once. The
                    # mean over segment outputs therefore also includes it once.
                    return jnp.mean(jax.vmap(lambda k,f:fn(k,inputs,f))(keys,fields),axis=0)
                pre=layer_mean(first,0,x,f0)
                logits=layer_mean(last,1,jax.nn.relu(pre),f1)
                return pre,logits
            return jax.vmap(one)(batches,input_batches,fields0,fields1)
        return jax.jit(run)

    @lru_cache(maxsize=4)
    def segmented_counts(self,segment_samples):
        counter=self.counts(segment_samples)
        def run(base_key,fields):
            keys=jax.vmap(lambda s:jax.random.fold_in(base_key,s))(jnp.arange(fields.shape[0]))
            return jax.vmap(counter)(keys,fields)
        return jax.jit(run)
