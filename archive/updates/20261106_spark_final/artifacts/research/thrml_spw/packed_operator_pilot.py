"""Large actual THRML operator programmed only from a packed control artifact.

No original converted target weight tensor is loaded by this process. Its external
FP32 arrays are observer inputs and archived reference/output evidence, not weights.
"""
import json,time
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from .packed_native import PackedNativeLinear
from .verify_transformer_control import sha
from .provenance import manifest


def main():
    if jax.default_backend()!='gpu':raise RuntimeError('This pinned large comparison requires CUDA')
    artifact=Path('results/thrml_spw/packed_quarter_controls/stage11.npz')
    exported=json.loads(Path('results/thrml_spw/packed_quarter_controls/REPORT.json').read_text())
    assert sha(artifact)==exported['artifact_sha256'][str(artifact)]
    source=Path('results/thrml_spw/transformer_quarter/runs/validation_T512_seed0')
    original=json.loads((source/'stage11_native_output.json').read_text());signature=original['signature']
    assert signature['samples']==512 and signature['seed']==0 and signature['stage']==11 and signature['domain']==970730
    xp,rp,yp=[source/f'stage11_{name}.npy' for name in ['inputs','conditional_reference','native_output']]
    assert sha(xp)==signature['inputs_sha256'] and sha(rp)==signature['reference_sha256'] and sha(yp)==original['output_sha256']
    out=Path('results/thrml_spw/packed_operator_pilot');out.mkdir(exist_ok=True)
    started=time.monotonic();layer=PackedNativeLinear(artifact);bank=layer.bank
    x,reference,direct=np.load(xp),np.load(rp),np.load(yp);chunks=[];fn=layer.compiled(512)
    for batch,first in enumerate(range(0,len(x),8)):
        key=jax.random.key(970730)
        for value in [0,11,batch]:key=jax.random.fold_in(key,value)
        chunk=x[first:first+8];y=fn(key,jnp.asarray(chunk.reshape(-1,chunk.shape[-1])))
        chunks.append(np.asarray(y).reshape(chunk.shape[0],chunk.shape[1],-1))
    y=np.concatenate(chunks);np.save(out/'native_output.npy',y)
    np.testing.assert_allclose(y,direct,rtol=2e-5,atol=2e-5)
    delta=y.astype(np.float64)-direct
    scaled=float(np.max(np.abs(delta))/(1+np.max(np.abs(direct))));assert scaled<2e-5
    t,batches=512,(len(x)+7)//8;free=len(bank.free_indices)
    assert free==original['controls']['free_pbits']
    assert batches*t*free==original['free_node_updates']
    assert batches*t*bank.logical_pbits==original['logical_weight_readouts']
    assert x.shape[0]*x.shape[1]*bank.logical_pbits*t==original['binary_product_terms']
    variance=layer.span**2*bank.realized_probabilities*(1-bank.realized_probabilities)
    predicted=float(np.mean(np.mean(x.astype(np.float64)**2,axis=(0,1))@variance)/t)
    np.testing.assert_allclose(predicted,original['predicted_sampling_mse'],rtol=1e-12)
    error=float(np.mean((y.astype(np.float64)-reference)**2))
    result={'passed':True,'projection':'transformer.h.2.mlp.c_proj','weight_count':bank.logical_pbits,
        'input_shape':list(x.shape),'samples':512,'original_target_FP32_weight_matrix_loaded':False,
        'controls':layer.controls(),'program_artifact':str(artifact),'program_artifact_sha256':sha(artifact),
        'input_sha256':sha(xp),'reference_sha256':sha(rp),'canonical_output_sha256':sha(yp),'output_sha256':sha(out/'native_output.npy'),
        'same_key_schedule':{'domain':970730,'seed':0,'stage':11,'contexts_per_bank':8},
        'exact_output_equality':bool(np.array_equal(y,direct)),'max_absolute_output_difference':float(np.max(np.abs(delta))),
        'scaled_max_output_difference':scaled,'rms_output_difference':float(np.sqrt(np.mean(delta**2))),
        'conditional_linear_mse':error,'predicted_sampling_mse':predicted,
        'free_node_evaluations_each':batches*t*free,'logical_weight_readouts_each':batches*t*bank.logical_pbits,
        'binary_product_terms_each':int(x.shape[0]*x.shape[1]*bank.logical_pbits*t),
        'elapsed_seconds_including_init_and_audit':time.monotonic()-started,
        'backend':jax.default_backend(),'devices':[str(d) for d in jax.devices()],
        'source_sha256':sha(__file__),'native_loader_sha256':sha(Path(__file__).with_name('packed_native.py')),
        'scope':'Actual official THRML large-operator execution from control codes/modes/shared ranges, no original target matrices loaded. Same existing student inputs/keys/counts, not new-study text, full stochastic packed-model quality, runtime compression, physical TSU computation or energy/latency proof.'}
    (out/'REPORT.json').write_text(json.dumps(result,indent=2)+'\n')
    manifest(out/'manifest.json',{'artifact':str(artifact),'source_inputs':str(xp),'samples':512},result)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
