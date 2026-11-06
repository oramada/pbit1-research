"""Actual native same-path partial-sum experiment on a large student projection."""
import json,subprocess,time
from pathlib import Path
import numpy as np
from .gpu_environment import cuda_environment
from .transformer_block_gpu import sha


def main():
    source=Path('results/thrml_spw/transformer_quarter/runs/validation_T256_seed0')
    out=Path('results/thrml_spw/partial_operator_pilot');out.mkdir(exist_ok=True)
    original=json.loads((source/'stage11_native_output.json').read_text());request={
        'weights':'results/thrml_spw/transformer_quarter/weights.npz','inputs':str(source/'stage11_inputs.npy'),'reference':str(source/'stage11_conditional_reference.npy'),
        'output':str(out/'native_output.npy'),'stage':11,'samples':256,'seed':0,'domain':970730,'contexts_per_batch':8,'field_bits':8,'field_bound':4.}
    worker=subprocess.Popen([str(Path('.venv-thrml-gpu/bin/python').absolute()),'-u','-m','research.thrml_spw.block_partial_worker'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,env=cuda_environment())
    try:
        assert json.loads(worker.stdout.readline())['status']=='ready'
        worker.stdin.write(json.dumps(request)+'\n');worker.stdin.flush();response=json.loads(worker.stdout.readline());assert response['status']=='ok',response
    finally:
        if worker.poll() is None:worker.stdin.write('{"action":"quit"}\n');worker.stdin.flush();worker.wait(timeout=60)
        worker.stdin.close();worker.stdout.close()
    meta=json.loads(Path(response['metadata']).read_text())
    for k in ['weights_sha256','inputs_sha256','reference_sha256','samples','seed','stage','domain','field_bits','field_bound']:assert meta['signature'][k]==original['signature'][k]
    for k in ['free_node_updates','logical_weight_readouts','binary_product_terms']:assert meta[k]==original[k]
    direct=np.load(source/'stage11_native_output.npy');partial=np.load(out/'native_output.npy')
    delta=partial.astype(np.float64)-direct
    scaled=float(np.max(np.abs(delta))/(1+np.max(np.abs(direct))));assert scaled<2e-5
    ledger=meta['observer_operation_ledger_per_full_bank_batch']
    report={'passed':True,'projection':'transformer.h.2.mlp.c_proj','weight_count':2359296,'input_shape':meta['input_shape'],'samples':256,
        'max_absolute_output_difference':float(np.max(np.abs(delta))),'scaled_max_output_difference':scaled,'rms_output_difference':float(np.sqrt(np.mean(delta**2))),
        'free_node_evaluations_each':meta['free_node_updates'],'binary_product_terms_each':meta['binary_product_terms'],
        'modeled_scale_multiply_reduction_factor':ledger['direct_sample_span_multiplications']/ledger['partial_postaccumulation_span_multiplications'],
        'accumulator_size_increase_factor':ledger['partial_output_accumulator_float32_values']/ledger['direct_output_accumulator_float32_values'],
        'operation_ledger_per_bank_batch':ledger,'source_sha256':sha(__file__),
        'scope':'Actual official THRML large-operator same-key comparison on preserved stochastic student activations. Algebraic scaling and accumulator tradeoff, not end-to-end model quality, measured latency/energy, or hardware proof.'}
    (out/'REPORT.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
