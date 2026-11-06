"""Independent same-path output and arithmetic ledger check for the partial pilot."""
import json
from pathlib import Path
import numpy as np
from .verify_transformer_control import sha,quantize


def main():
    out=Path('results/thrml_spw/partial_operator_pilot');source=Path('results/thrml_spw/transformer_quarter/runs/validation_T256_seed0')
    meta=json.loads((out/'native_output.json').read_text());old=json.loads((source/'stage11_native_output.json').read_text());report=json.loads((out/'REPORT.json').read_text())
    for path,digest in meta['signature']['source_sha256'].items():assert sha(path)==digest
    for key in ['weights_sha256','inputs_sha256','reference_sha256','samples','seed','stage','domain','field_bits','field_bound']:assert meta['signature'][key]==old['signature'][key]
    assert sha(out/'native_output.npy')==meta['output_sha256'];assert sha(source/'stage11_native_output.npy')==old['output_sha256']
    x=np.load(source/'stage11_inputs.npy');ref=np.load(source/'stage11_conditional_reference.npy');direct=np.load(source/'stage11_native_output.npy');partial=np.load(out/'native_output.npy')
    assert sha(source/'stage11_inputs.npy')==meta['signature']['inputs_sha256'];assert sha(source/'stage11_conditional_reference.npy')==meta['signature']['reference_sha256']
    master=Path('results/thrml_spw/transformer_quarter/weights.npz');assert sha(master)==meta['signature']['weights_sha256'];weights=np.load(master)
    weight=weights['weight_11'];meanw,var,free,clipped=quantize(weight,8,4.);mu=x.astype(np.float64)@meanw+weights['bias_11']
    bias=float(np.mean((mu-ref)**2));prediction=float(np.mean(np.mean(x.astype(np.float64)**2,axis=(0,1))@var)/256)
    for value,k in [(bias,'conditional_control_bias_mse'),(prediction,'predicted_sampling_mse'),(prediction+bias,'predicted_total_mse'),(float(np.mean((partial.astype(np.float64)-ref)**2)),'conditional_linear_mse')]:np.testing.assert_allclose(value,meta[k],rtol=1e-9,atol=1e-14)
    n=weight.size;rows=8*x.shape[1];groups=weight.shape[0]//128;outputs=weight.shape[1];ledger=meta['observer_operation_ledger_per_full_bank_batch']
    checks={'binary_product_terms':rows*n*256,'direct_sample_span_multiplications':n*256,'partial_postaccumulation_span_multiplications':rows*groups*outputs,
        'direct_expanded_lower_product_terms':rows*n,'grouped_lower_product_terms':rows*groups*outputs,'direct_output_accumulator_float32_values':rows*outputs,'partial_output_accumulator_float32_values':rows*groups*outputs}
    for key,value in checks.items():assert ledger[key]==value
    for key in ['free_node_updates','logical_weight_readouts','binary_product_terms']:assert meta[key]==old[key]
    assert meta['free_node_updates']==2*256*free
    delta=partial.astype(np.float64)-direct;np.testing.assert_allclose(np.sqrt(np.mean(delta**2)),report['rms_output_difference'],rtol=1e-12)
    assert np.max(np.abs(delta))/(1+np.max(np.abs(direct)))<2e-5
    fraction=float(np.sqrt(np.mean(delta**2))/np.sqrt(old['conditional_linear_mse']))
    result={'passed':True,'source_sha256':sha(__file__),'weight_count':n,'same_key_and_draw_count':True,'scaled_max_output_difference':report['scaled_max_output_difference'],
        'observer_difference_fraction_of_native_error_RMS':fraction,'scale_multiply_reduction_factor':n*256/(rows*groups*outputs),'group_accumulator_increase_factor':groups,
        'scope':'Independent raw output, source/key/input, conditional MSE and algebraic resource audit; single native operator, not end-to-end quality or physical speed/energy.'}
    (out/'evidence_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
    text=f'''# Native grouped partial-sum operator pilot

A real 2,359,296-weight MLP-down projection in Transformer block2 was executed
through official THRML using the same finite controls, current student inputs,
keys and T256 as the original observer. A new observer accumulates binary group
partial sums first, then applies shared range scaling after temporal averaging.
The weight probabilities and stochastic update/read counts are unchanged.

Scaled maximum output difference is {report['scaled_max_output_difference']:.3g};
RMS difference is {report['rms_output_difference']:.3g}, or {fraction:.3g} of the
original native stochastic-error RMS. Float32 summation order differs. Tiny CPU
and CUDA checks additionally cover ragged groups and exact singleton clamps.

For each eight-context/65-token bank batch, the operation ledger changes range
scaling from {n*256:,} multiplications to {rows*groups*outputs:,}, about
{result['scale_multiply_reduction_factor']:.2f} times fewer. Grouping also reduces
expanded lower-correction product terms by128. Binary product terms and all
stochastic draws remain identical. The accumulator grows from {rows*outputs:,}
to {rows*groups*outputs:,} float32 values, a24-fold increase. The latter is
{rows*groups*outputs*4:,} bytes. Actual memory traffic, kernel fusion, initialization,
readout and hardware organization require their own measurements.

This is an actual native large-operator arithmetic-equivalence pilot and resource
tradeoff. It does not demonstrate end-to-end neural quality for the new observer,
physical TSU computation, measured speed/energy improvement or system superiority.
The canonical three-block neural experiment continues using its unchanged observer.

Raw evidence: partial_operator_pilot/REPORT.json, native_output.npy/.json and
evidence_audit.json, linked to the preserved quarter-model student arrays.
'''
    text=text.replace('block2','block 2').replace('by128','by 128').replace('a24-fold','a 24-fold')
    Path('results/thrml_spw/PARTIAL_OPERATOR_REPORT.md').write_text(text)

if __name__=='__main__':main()
