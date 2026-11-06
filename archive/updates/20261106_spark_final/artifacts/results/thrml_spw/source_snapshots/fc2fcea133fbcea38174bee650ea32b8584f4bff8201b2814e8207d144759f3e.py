"""Independent packed-large-operator audit, without original target matrices."""
import json
from pathlib import Path
import numpy as np
from .verify_transformer_control import sha


def main():
    out=Path('results/thrml_spw/packed_operator_pilot');r=json.loads((out/'REPORT.json').read_text())
    source=Path('results/thrml_spw/transformer_quarter/runs/validation_T512_seed0');canonical=json.loads((source/'stage11_native_output.json').read_text())
    assert not r['original_target_FP32_weight_matrix_loaded'] and r['backend']=='gpu'
    artifact=Path(r['program_artifact']);assert sha(artifact)==r['program_artifact_sha256'];assert sha(out/'native_output.npy')==r['output_sha256']
    x,ref,direct=[np.load(source/f'stage11_{name}.npy') for name in ['inputs','conditional_reference','native_output']];y=np.load(out/'native_output.npy')
    for path,key in [(source/'stage11_inputs.npy','input_sha256'),(source/'stage11_conditional_reference.npy','reference_sha256'),(source/'stage11_native_output.npy','canonical_output_sha256')]:assert sha(path)==r[key]
    with np.load(artifact,allow_pickle=False) as packed:
        header=json.loads(packed['header'].tobytes().decode());shape=tuple(header['shape']);n=int(np.prod(shape))
        bits=np.unpackbits(packed['modes'],bitorder='little').reshape(-1,2)[:n]
        modes=bits[:,0]+2*bits[:,1];assert np.all(modes<3)
        code=packed['codes'];free=np.flatnonzero(modes==0);assert len(free)==len(code)
        field=(-4+code.astype(np.float64)*(8/255)).astype(np.float32)
        q=np.zeros(n,dtype=np.float64);q[modes==2]=1;q[free]=np.exp(-np.logaddexp(0,-2*field.astype(np.float64)));q=q.reshape(shape)
        span=np.repeat(packed['span'],128,axis=0)[:shape[0]]
    assert shape==(3072,768) and r['samples']==512 and r['same_key_schedule']=={'domain':970730,'seed':0,'stage':11,'contexts_per_bank':8}
    predicted=float(np.mean(np.mean(x.astype(np.float64)**2,axis=(0,1))@(span**2*q*(1-q)))/512)
    np.testing.assert_allclose(predicted,canonical['predicted_sampling_mse'],rtol=1e-12);np.testing.assert_allclose(predicted,r['predicted_sampling_mse'],rtol=1e-12)
    assert r['free_node_evaluations_each']==canonical['free_node_updates']==2*512*len(free)
    assert r['logical_weight_readouts_each']==canonical['logical_weight_readouts']==2*512*n
    assert r['binary_product_terms_each']==canonical['binary_product_terms']==16*65*512*n
    delta=y.astype(np.float64)-direct;assert r['exact_output_equality']==bool(np.array_equal(y,direct));assert np.max(np.abs(delta))==r['max_absolute_output_difference']
    np.testing.assert_allclose(np.mean((y.astype(np.float64)-ref)**2),r['conditional_linear_mse'],rtol=1e-12)
    assert sha('research/thrml_spw/packed_operator_pilot.py')==r['source_sha256'];assert sha('research/thrml_spw/packed_native.py')==r['native_loader_sha256']
    audit={'passed':True,'weight_count':n,'exact_output_equality':bool(np.array_equal(y,direct)),'free_node_evaluations':r['free_node_evaluations_each'],
        'original_target_matrix_loaded_by_auditor':False,'source_sha256':sha(__file__),
        'scope':'Raw packed bits decoded independently, exact output/checksum/key/counter and Bernoulli-variance checks on one real native projection; no full-model packed execution or physical hardware claim'}
    (out/'evidence_audit.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps(audit,indent=2))
    report=Path('results/thrml_spw/PACKED_CONTROLS_REPORT.md')
    marker='\n## Large native execution from the packed program\n'
    text=report.read_text().split(marker)[0]
    text+=marker+f'''
The2,359,296-weight block2 MLP-down projection was also executed through the actual
CUDA THRML sampler using only its packed controls and saved old student inputs.
It did not load the original converted FP32 weight tensor. At T512 and the same
keys/bank schedule, all output values match the canonical execution exactly.
Each path evaluates {r['free_node_evaluations_each']:,} free nodes. A separate audit
independently decodes mode bits, reconstructs stationary variance and verifies
checksums, keys, counts and raw outputs without target parameter tensors.

This extends the tiny tests to a real large stochastic operator. It remains a
single-operator result; it does not establish full stochastic packed-model quality,
compact sampler memory, silicon area or a speed/energy advantage. The ongoing
conservative calibration uses its unchanged canonical source representation.
Raw evidence: packed_operator_pilot/REPORT.json and evidence_audit.json.
'''
    report.write_text(text.replace('The2,359,296','The 2,359,296').replace('block2','block 2'))

if __name__=='__main__':main()
