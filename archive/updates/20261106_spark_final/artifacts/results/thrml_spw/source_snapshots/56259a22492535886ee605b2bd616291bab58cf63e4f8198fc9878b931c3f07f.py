"""Check one completed million-node operator while full study is still running."""
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from .calibrated_mnist_path_audit import PublicLayer
from .verify_transformer_control import sha


def main():
    out=Path('results/thrml_spw/calibrated_transformer');p=json.loads((out/'protocol.json').read_text());a=np.load(p['control_artifacts'][0]['path']);shape=a['codes'].shape
    folder=out/'runs/ideal_seed0';meta=json.loads((folder/'stage0_native_output.json').read_text());xp=folder/'stage0_inputs.npy';yp=folder/'stage0_native_output.npy'
    assert sha(xp)==meta['signature']['input_sha256'] and sha(yp)==meta['output_sha256']
    fields=(-4+a['codes'].reshape(-1).astype(float)*8/255).astype(np.float32);keys=[]
    for bank in range(8):
        key=jax.random.key(1140730)
        for v in [0,0,bank]:key=jax.random.fold_in(key,v)
        keys.append(key)
    layer=PublicLayer(np.full(shape,.5));assert not layer.clamped and layer.n_free==np.prod(shape)
    q=np.asarray(layer.frequencies(1024)(jnp.stack(keys),jnp.asarray(fields))).astype(float)
    lower=np.repeat(a['lower'],128,0);span=np.repeat(a['span'],128,0);x=np.load(xp).reshape(8,520,shape[0]).astype(float)
    expected=(x@(lower+span*q)+a['bias']).reshape(64,65,shape[1]);actual=np.load(yp)
    error=float(np.max(np.abs(actual.astype(float)-expected))/(1+np.max(np.abs(expected))));assert error<2e-5,error
    r={'passed':True,'stage':0,'seed':0,'condition':'ideal','weight_nodes':int(np.prod(shape)),'bank_samples':8,'T':1024,'scaled_output_error':error,'tolerance':2e-5,'source_sha256':sha(__file__),
        'scope':'A single completed large operator independently replayed while rest ofstudy runs. This is verification work only, not a new quality replicate or proof ofall180operators.'}
    (out/'large_operator_preflight.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))


if __name__=='__main__':main()
