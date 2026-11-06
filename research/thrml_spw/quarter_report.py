"""Three-block development results, retained failures and explicit resources."""
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

ROOT=Path('results/thrml_spw');OUT=ROOT/'transformer_quarter'

def read(path):
    with path.open() as f:return list(csv.DictReader(f))

def main():
    final=json.loads((OUT/'manifest.json').read_text());audit=json.loads((OUT/'evidence_audit.json').read_text());assert audit['passed']
    p=final['config'];selection=final['selection'];t=selection['selected_samples'];summaries=read(OUT/'summary.csv');base=read(OUT/'baseline_metrics.csv');prefix=read(OUT/'prefix_metrics.csv')
    fp={split:next(r for r in base if r['split']==split and r['condition']=='FP32') for split in ('validation','test')}
    selected=next(r for r in summaries if r['split']=='test')
    test_pass=bool(float(selected['cross_entropy_mean'])-float(fp['test']['cross_entropy'])<=np.log(1.05) and float(fp['test']['token_accuracy'])-float(selected['token_accuracy_mean'])<=.01)
    run=json.loads((OUT/'runs'/f'test_T{t}_seed0'/'run.json').read_text());controls=[json.loads(Path(s['metadata']).read_text())['controls'] for s in run['stages']]
    n=p['converted_matrix_weights'];free=sum(c['free_pbits'] for c in controls);groups=sum(c['groups'] for c in controls);bias=p['deterministic_converted_module_biases']
    codes=sum(c['hypothetical_packed_controller_bits'] for c in controls);modes=sum(c['hypothetical_endpoint_mode_bits'] for c in controls);ranges=sum(c['metadata_bits'] for c in controls);shared=sum(c['hypothetical_shared_range_and_beta_bits'] for c in controls)
    costs={'logical_weight_nodes':n,'free_nodes':free,'clamped_nodes':n-free,'local_groups':groups,'converted_fraction':p['converted_fraction'],
        'hypothetical_controller_bits':codes,'hypothetical_endpoint_mode_bits':modes,'range_bits':ranges,'shared_control_bits':shared,'neural_bias_bits':bias*32,
        'hypothetical_control_modes_range_bias_bits_per_weight':(codes+modes+ranges+shared+bias*32)/n,
        'actual_field_bytes':sum(c['bias_storage_bytes_simulator'] for c in controls),'actual_uint32_code_bytes':free*4,
        'expanded_range_buffer_bytes':sum(c['expanded_range_buffer_bytes_simulator'] for c in controls),
        'test_free_node_evaluations_per_seed':int(selected['free_node_updates']),'sampling_rounds_per_eight_context_bank_batch':12*t,
        'serial_sampling_rounds_for_test_bank_batches':12*t*4,
        'affine4_nominal_bits_per_weight':(4*n+32*groups+4*groups+bias*32)/n,
        'affine8_nominal_bits_per_weight':(8*n+32*groups+8*groups+bias*32)/n,
        'scope':'Digital code ledgers and actual simulator arrays are separate; neither measures physical area, energy or latency. Subtotals exclude stochastic state/program/accumulator and all unconverted parameters.'}
    (OUT/'resource_ledger.json').write_text(json.dumps(costs,indent=2)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for split,color in [('validation','tab:blue'),('test','tab:orange')]:
        rows=sorted([r for r in summaries if r['split']==split],key=lambda r:int(r['samples']))
        axes[0].errorbar([int(r['samples']) for r in rows],[float(r['relative_perplexity']) for r in rows],
            yerr=[[float(r['relative_perplexity'])-float(r['relative_perplexity_lower']) for r in rows],[float(r['relative_perplexity_upper'])-float(r['relative_perplexity']) for r in rows]],fmt='o-',color=color,label=split+' development')
    axes[0].axhline(1.05,color='black',ls=':',label='5% calibration tolerance');axes[0].set(xscale='log',xlabel='Samples per operator',ylabel='Perplexity / FP32',title='21.23M native weights across three blocks');axes[0].legend(fontsize=7);axes[0].set_xticks([256,512]);axes[0].set_xticklabels([256,512]);axes[0].xaxis.set_minor_formatter(NullFormatter())
    for time in sorted({int(r['samples']) for r in prefix if r['split']=='validation'}):
        values=[]
        for stage in [0,4,8,12]:
            rows=[r for r in prefix if r['split']=='validation' and int(r['samples'])==time and int(r['converted_prefix'])==stage]
            values.append(np.exp(np.mean([float(r['cross_entropy']) for r in rows])-float(fp['validation']['cross_entropy'])))
        axes[1].plot(range(4),values,'o-',label=f'T={time}')
    axes[1].set(xticks=range(4),xticklabels=['FP32','+block0','+block1','+block2'],ylabel='Calibration perplexity / FP32',title='Prefix effects, conditional on forward order');axes[1].legend(fontsize=7)
    for ax in axes:ax.grid(axis='y',alpha=.2)
    fig.savefig(ROOT/'transformer_quarter_evidence.png',dpi=180);plt.close(fig)
    lines=['# Native conversion of the first three Transformer blocks\n',
        f"Twelve actual DistilGPT-2 projection matrices run through official THRML: {n:,} normalized weights, {100*p['converted_fraction']:.4f}% of model parameters. Each normalized weight has one logical p-bit. Local input-block128 affine lower/span metadata and8-bit uniform dimensionless Ising-b controls in [-4,4] are explicit. Biases, LayerNorm, embeddings/head and the last three blocks stay deterministic.\n",
        'Every later student input is regenerated after all earlier native substitutions, including block boundaries. Twelve operators accumulate linear outputs before downstream nonlinearities. A tiny two-layer causal replay check separately verifies matrix orientation, exact same-output replay, changed student activations after an upstream substitution and restored original weights.\n',
        '## Development protocol and results\n',
        'The predeclared ascending search is T256/512/1024, five native calibration seeds per budget. The first mean CE increase<=log(1.05) and token-accuracy loss<=one point is selected. Original test windows provide a development diagnostic at selected T; no test scores choose T. The completed first-block confirmation text is never loaded or tuned on. All failed budgets are retained.\n',
        '| Split | T per operator | Perplexity | Relative to FP32 | Conditional MC interval |\n|---|---:|---:|---:|---:|']
    for r in summaries:lines.append(f"| {r['split']} | {r['samples']} | {float(r['perplexity']):.3f} | {float(r['relative_perplexity']):.4f} | [{float(r['relative_perplexity_lower']):.4f},{float(r['relative_perplexity_upper']):.4f}] |")
    lines.extend([f"\nCalibration gate passed: {selection['gate_passed']}; selected T={t}. Test development perplexity is {float(selected['perplexity']):.3f} versus FP32 {float(fp['test']['perplexity']):.3f}. This development test outcome {'meets' if test_pass else 'misses'} the same tolerance. Calibration passed narrowly; this separate outcome is retained without changing T. These are five Monte Carlo seeds of one frozen model on fixed reused text, not confirmation or corpus-population evidence. Prefix effects describe this forward-order experiment, not isolated sensitivity rankings.\n",
        '## Conventional controls and programmed mean\n',
        '| Split | Condition | Perplexity |\n|---|---|---:|'])
    for r in base:lines.append(f"| {r['split']} | {r['condition']} | {float(r['perplexity']):.3f} |")
    lines.extend(['\nThe programmed-mean diagnostic evaluates the stationary expected weights induced by the actual8-bit lattice without sampling. It separates persistent control error from finite-sample degradation; it is conventional deterministic inference, not another native sample result. Affine4/8 preserve zero and use input blocks128; these are numerical dequantized controls, not optimized packed kernels or exhaustive best-available quantizers.\n',
        '## Resources and limitations\n',
        f"The selected test schedule evaluates {int(selected['free_node_updates']):,} free nodes per seed over four eight-context bank batches. Depth is {12*t:,} sampling rounds per bank batch, or {12*t*4:,} under serial bank reuse for the whole diagnostic. Logical reads also count explicit endpoints. Neither updates nor rounds are measured physical transitions or chip timing.\n",
        f"There are {free:,} free nodes and {n-free:,} explicit endpoint clamps. Range metadata is {ranges//8:,} bytes (0.5 bits/weight). The hypothetical digital control/mode/range/bias ledger totals {costs['hypothetical_control_modes_range_bias_bits_per_weight']:.4f} bits per converted matrix weight, compared with nominal affine4 {costs['affine4_nominal_bits_per_weight']:.4f} and affine8 {costs['affine8_nominal_bits_per_weight']:.4f}. Endpoint mode uses a conservative two-bit code per node. Other layouts require their own accounting.\n",
        f"Actual simulator arrays include {costs['actual_field_bytes']:,} bytes of float32 fields, {costs['actual_uint32_code_bytes']:,} bytes of uint32 codes and {costs['expanded_range_buffer_bytes']:,} bytes of expanded lower/span arrays. Stochastic states, probability arrays, program objects, accumulators, activations and unconverted model parameters add costs. This is not achieved software compression or physical-area measurement. Observer multiplication and accumulation remain modeled external arithmetic.\n",
        f"Independent audits cover {audit['native_runs']} complete native cases, {audit['operator_checkpoints']} native operator checkpoints, separate pre/post-hook replay of every final token array and eight independently reproduced conventional/mean model conditions. Every input/reference/output and source/key lineage, finite-control bias/variance decomposition, count, gate and raw token score is checked.\n",
        'A new128-window confirmation set has been reserved with no overlap with either consumed native text set or the documented older pilot, and no confirmation scores have been computed. Its plan freezes the development-selected budget and controls before evaluation. The goal remains active for that confirmation, broader conversion, grouped partial-sum operators, nonideal large devices, training and physical-system costs. A quarter-model quality result does not establish full-model or hardware superiority.\n'])
    text='\n'.join(lines)+'\n'
    for a,b in {'metadata and8-bit':'metadata and 8-bit','actual8-bit':'actual 8-bit','input blocks128':'input blocks of 128','new128-window':'new 128-window'}.items():text=text.replace(a,b)
    (ROOT/'TRANSFORMER_QUARTER_REPORT.md').write_text(text)
    print(json.dumps({'selection':selection,'test':selected,'resources':costs},indent=2))

if __name__=='__main__':main()
