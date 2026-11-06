"""Report actual finite-bias native inference and hypothetical control code cost."""
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path('results/thrml_spw');OUT=ROOT/'transformer_control'

def read(path):
    with path.open() as f:return list(csv.DictReader(f))

def main():
    final=json.loads((OUT/'manifest.json').read_text());selection=final['selection'];audit=json.loads((OUT/'evidence_audit.json').read_text());assert audit['passed']
    summaries=read(OUT/'summary.csv');expected=read(OUT/'expected_metrics.csv');base=read(OUT/'baseline_metrics.csv');bits=selection['selected_bits']
    fp={r['split']:float(r['cross_entropy']) for r in base}
    costs=[];decomp=[]
    for r in read(OUT/'native_metrics.csv'):
        split,b,seed=r['split'],int(r['field_bits']),int(r['sampling_seed'])
        folder=OUT/'runs'/f'{split}_B{b}_T256_seed{seed}';run=json.loads((folder/'run.json').read_text())
        for stage in run['stages']:
            meta=json.loads(Path(stage['metadata']).read_text());ctrl=meta['controls']
            decomp.append({'split':split,'field_bits':b,'sampling_seed':seed,'stage':stage['stage'],**{k:meta[k] for k in ['conditional_linear_mse','conditional_control_bias_mse','predicted_sampling_mse','predicted_total_mse','sampling_mse_vs_programmed_expectation']}})
        if split!='test' or seed!=0:continue
        controls=[json.loads(Path(s['metadata']).read_text())['controls'] for s in run['stages']]
        free=sum(c['free_pbits'] for c in controls);n=sum(c['logical_pbits'] for c in controls)
        codes=sum(c['hypothetical_packed_controller_bits'] for c in controls)
        modes=sum(c['hypothetical_endpoint_mode_bits'] for c in controls)
        ranges=sum(c['metadata_bits'] for c in controls);shared=sum(c['hypothetical_shared_range_and_beta_bits'] for c in controls)
        bias=final['config']['deterministic_converted_module_biases']*32
        costs.append({'field_bits':b,'logical_pbits':n,'free_nodes':free,'clamped_nodes':n-free,'hypothetical_packed_controller_bits':codes,
            'hypothetical_endpoint_mode_bits':modes,'range_bits':ranges,'shared_control_bits':shared,'neural_bias_bits':bias,
            'declared_packed_control_modes_ranges_bias_bits':codes+modes+ranges+shared+bias,
            'hypothetical_bits_per_matrix_weight':(codes+modes+ranges+shared+bias)/n,
            'actual_float32_field_bytes':sum(c['bias_storage_bytes_simulator'] for c in controls),
            'actual_uint32_code_array_bytes':free*4,'range_clipped_fields':sum(c['range_clipped_fields'] for c in controls),
            'scope':'Hypothetical packed digital control bookkeeping, excluding stochastic states/programs/accumulators, not actual Python memory or analog area; endpoint mode is a conservative explicit 2-bit code per logical node'})
    for name,rows in [('controller_costs.csv',costs),('conditional_mse_decomposition.csv',decomp)]:
        with (OUT/name).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    conventional=next(r for r in read(ROOT/'block_quantization_controls/metrics.csv') if r['condition']=='affine_zero_block128_4' and r['split']=='test')
    conventional_bits=(sum(int(conventional[k]) for k in ('nominal_code_bits','nominal_scale_bits','nominal_zero_point_bits'))+final['config']['deterministic_converted_module_biases']*32)/7077888
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for split,color in [('validation','tab:blue'),('test','tab:orange')]:
        native=sorted([r for r in summaries if r['split']==split],key=lambda r:int(r['field_bits']))
        mean=sorted([r for r in expected if r['split']==split],key=lambda r:int(r['field_bits']))
        axes[0].plot([int(r['field_bits']) for r in native],[float(r['relative_perplexity']) for r in native],'o-',color=color,label=f'{split}: native T256')
        axes[0].plot([int(r['field_bits']) for r in mean],[np.exp(float(r['cross_entropy'])-fp[split]) for r in mean],'x--',color=color,label=f'{split}: programmed mean')
    axes[0].axhline(1.05,color='black',ls=':',label='Calibration tolerance')
    axes[0].set(yscale='log',xlabel='Bits per finite Ising-b control',ylabel='Perplexity / FP32',title='Bias quantization versus sampling error',xticks=[4,6,8,10,12]);axes[0].legend(fontsize=7)
    axes[1].bar([str(c['field_bits']) for c in costs],[c['hypothetical_bits_per_matrix_weight'] for c in costs],color='tab:purple')
    axes[1].axhline(conventional_bits,color='black',ls=':',label=f'Affine4 block128 control: {conventional_bits:.3f} bits/weight')
    axes[1].set(xlabel='Controller bits (test conditions)',ylabel='Hypothetical digital bits / matrix weight',title='Control, endpoint modes, ranges and biases');axes[1].legend(fontsize=7)
    for ax in axes:ax.grid(axis='y',alpha=.2)
    fig.savefig(ROOT/'transformer_control_evidence.png',dpi=180);plt.close(fig)
    selected=next(r for r in summaries if r['split']=='test' and int(r['field_bits'])==bits);cost=next(c for c in costs if c['field_bits']==bits)
    lines=['# Finite-bias controller experiment in the native Transformer block\n',
        'The four first-block projections (7,077,888 weights, 8.64% of DistilGPT-2) use official THRML with input-block128 affine ranges and regenerated student activations. Each normalized weight still has one logical p-bit. This extends the existing representation to finite programming resolution.\n',
        '## Protocol and scope\n',
        'A predeclared ascending 4/6/8/10/12-bit search uses five native calibration seeds at fixed T256. Every finite field is one of 2^B uniform dimensionless Ising-b levels in [-4,4]. The first mean relative-perplexity increase <=5% and token-accuracy loss <=1 point is selected. The test evaluates selected bits and the predeclared 4-bit diagnostic, five seeds each. No test labels select precision. Bound 4 is a design assumption, not an Extropic voltage specification; failures apply to this lattice and budget, not every possible low-bit controller. These text windows are reused development data, not fresh confirmatory evidence.\n',
        '| Split | Control bits | Native T256 perplexity | Relative to FP32 |\n|---|---:|---:|---:|']
    for r in summaries:lines.append(f"| {r['split']} | {r['field_bits']} | {float(r['perplexity']):.3f} | {float(r['relative_perplexity']):.4f} |")
    lines.extend([f"\nCalibration selected {bits} bits; gate passed: {selection['gate_passed']}. Test perplexity is {float(selected['perplexity']):.3f}, relative {float(selected['relative_perplexity']):.4f} with approximate Monte Carlo interval [{float(selected['relative_perplexity_lower']):.4f},{float(selected['relative_perplexity_upper']):.4f}]. Intervals condition on this frozen model and fixed corpus windows. The point estimate is inside 5%, while the interval extends slightly above it; this is not a 95% guarantee of staying within that tolerance.\n",
        '## Persistent controller error\n',
        'An independent programmed-mean control uses the exact stationary probabilities induced by quantized fields in all four matrices, without sampling. It isolates the error that remains as T tends to infinity. It is a diagnostic conventional model, not another THRML sampling result. Every expected-control token array is independently reproduced through the real pretrained model.\n',
        '| Split | Control bits | Programmed-mean perplexity |\n|---|---:|---:|'])
    for r in expected:lines.append(f"| {r['split']} | {r['field_bits']} | {float(r['perplexity']):.3f} |")
    lines.extend(['\nFor each actual student input, the worker and independent auditor compute the conditional control-bias MSE and Bernoulli variance/T separately. Predicted total MSE is their sum, not variance/T alone. Raw conditional_mse_decomposition.csv preserves realized error and predictions, including negative low-bit conditions.\n',
        '## Representation and costs\n',
        f"The selected code model has {cost['free_nodes']:,} free and {cost['clamped_nodes']:,} explicitly clamped logical nodes. A hypothetical packed digital ledger counts {cost['hypothetical_packed_controller_bits']:,} control bits, {cost['hypothetical_endpoint_mode_bits']:,} endpoint-mode bits, {cost['range_bits']:,} local-range bits, {cost['shared_control_bits']:,} shared bound/beta bits and {cost['neural_bias_bits']:,} retained neural-bias bits: {cost['hypothetical_bits_per_matrix_weight']:.4f} bits per matrix weight for this subtotal, before stochastic states, program objects and accumulators. The endpoint layout uses a conservative explicit two-bit mode per node; other layouts require their own accounting.\n",
        f"Actual software still stores {cost['actual_float32_field_bytes']:,} bytes of float32 fields plus {cost['actual_uint32_code_array_bytes']:,} bytes of uint32 codes, expanded lower/span arrays and program/probability/state/accumulator buffers. The packed ledger is not achieved Python compression. An analog retained-bias controller could have a different area model; this simulator does not establish it.\n",
        f'The conventional affine4 packed code/scale/zero-point/bias ledger is {conventional_bits:.4f} bits per selected matrix weight. The previous floating-field grouped T256 test scored 116.354 versus FP32 112.207. The conventional zero-preserving affine4 block128 control scores 112.806. Finite control feasibility is not a quality, memory, energy or latency win over that control. CUDA is software acceleration, and observer multiplication/accumulation is external modeled arithmetic.\n',
        f"Verification: {audit['native_runs']} complete native runs, {audit['native_operators']} raw operator checkpoints, exact independent hook replay of all native cases and ten independently reproduced programmed-mean model conditions. CPU/CUDA same-path finite-control tests pass. The long-horizon goal remains active for controller-range optimization, fresh data, larger converted fractions, training and physical-system evidence.\n"])
    (ROOT/'TRANSFORMER_CONTROL_REPORT.md').write_text('\n'.join(lines)+'\n');print(json.dumps({'selected_test':selected,'selected_cost':cost},indent=2))

if __name__=='__main__':main()
