"""Compare completed native range-localization experiments and declared costs."""
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path('results/thrml_spw')

def read(path):
    with path.open() as f:return list(csv.DictReader(f))

def main():
    records=[];resources=[]; raw_by_group={}
    for name,directory in [('per_output','transformer_block0_gpu'),('input_block128','transformer_block0_group128')]:
        out=ROOT/directory
        manifest=json.loads((out/'manifest.json').read_text())
        audit=json.loads((out/'evidence_audit.json').read_text());assert audit['passed']
        selected=manifest['selection']['selected']['samples']
        raw_by_group[name]=read(out/'native_metrics.csv')
        sample=json.loads((out/'runs'/f'test_T{selected}_seed0'/'run.json').read_text())
        meta=[json.loads(Path(s['metadata']).read_text()) for s in sample['stages']]
        field=sum(m['controls']['bias_storage_bytes_simulator'] for m in meta)
        ranges=sum(m['controls']['metadata_bits'] for m in meta)//8
        expanded=sum(m['controls'].get('expanded_range_buffer_bytes_simulator',0) for m in meta)
        biases=manifest['config']['deterministic_converted_module_biases']*4
        resources.append({'grouping':name,'selected_samples':selected,'logical_pbits':7077888,
            'free_nodes':sum(m['free_node_updates']//(m['bank_batches']*selected) for m in meta),
            'range_bytes':ranges,'range_bits_per_weight':ranges*8/7077888,'field_bytes':field,
            'bias_bytes':biases,'declared_field_range_bias_bytes':field+ranges+biases,
            'expanded_range_buffers_bytes_extra':expanded,
            'test_free_node_updates':sample['free_node_updates'],
            'sampling_depth':4*selected,'gate_passed':manifest['selection']['gate_passed_on_calibration']})
        for row in read(out/'summary.csv'):
            records.append({'grouping':name,**row})
    for filename,rows in [('grouped_block_comparison.csv',records),('grouped_block_resources.csv',resources)]:
        with (ROOT/filename).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    paired=[]
    for split in ('validation','test'):
        common=set(int(r['samples']) for r in raw_by_group['per_output'] if r['split']==split)&set(int(r['samples']) for r in raw_by_group['input_block128'] if r['split']==split)
        for t in sorted(common):
            delta=[]
            for seed in range(5):
                left,right=[next(r for r in raw_by_group[g] if r['split']==split and int(r['samples'])==t and int(r['sampling_seed'])==seed) for g in ('per_output','input_block128')]
                delta.append(float(right['cross_entropy'])-float(left['cross_entropy']))
            delta=np.array(delta)
            paired.append({'split':split,'samples':t,'group128_minus_output_mean_CE':float(delta.mean()),'paired_MC_ci95_margin':float(1.96*delta.std(ddof=1)/np.sqrt(5)),'scope':'Paired seed/domain comparison; different clamp layouts change RNG-to-weight assignment, not identical weight trajectories'})
    with (ROOT/'grouped_block_paired.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(paired[0]));writer.writeheader();writer.writerows(paired)
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for group,color in [('per_output','tab:orange'),('input_block128','tab:blue')]:
        for split,style in [('validation','o-'),('test','s--')]:
            rows=sorted([r for r in records if r['grouping']==group and r['split']==split],key=lambda r:int(r['samples']))
            axes[0].plot([int(r['samples']) for r in rows],[float(r['relative_perplexity']) for r in rows],style,color=color,label=f'{group}: {split}')
    axes[0].axhline(1.05,color='black',ls=':',label='Calibration tolerance')
    axes[0].set(xscale='log',yscale='log',xlabel='Samples per operator',ylabel='Perplexity / FP32',title='Actual native four-projection inference')
    axes[0].legend(fontsize=7)
    axes[1].bar([r['grouping'] for r in resources],[r['test_free_node_updates']/1e9 for r in resources],color=['tab:orange','tab:blue'])
    axes[1].set(ylabel='Free-node evaluations / test seed (billions)',title='Calibration-selected schedules')
    for ax in axes:ax.grid(axis='y',alpha=.2)
    fig.savefig(ROOT/'grouped_block_comparison.png',dpi=180);plt.close(fig)
    a,b=resources
    grouped=next(r for r in records if r['grouping']=='input_block128' and r['split']=='test' and int(r['samples'])==b['selected_samples'])
    lines=['# Native Transformer range-localization experiment\n',
        'All four first-block matrix projections run through official THRML, with later student inputs regenerated after earlier replacements. Both encodings retain one logical p-bit per normalized matrix weight. This experiment compares affine local ranges; the prior direct p=w scalar and positive-network results remain separate.\n',
        'The new protocol predeclares input blocks of 128 per output, the same ascending T=64/256/1024 search, five sampling seeds and the same 5% perplexity/one-point token-accuracy calibration gate. Test scores do not select T. The corpus windows were already inspected in earlier research: this is a development comparison, not fresh confirmatory held-out evidence.\n',
        '| Grouping | Split | T | Perplexity | Relative to FP32 |\n|---|---|---:|---:|---:|']
    for r in records:lines.append(f"| {r['grouping']} | {r['split']} | {r['samples']} | {float(r['exp_mean_cross_entropy']):.3f} | {float(r['relative_perplexity']):.4f} |")
    lines.extend([f"\nThe new calibration-selected schedule is T={b['selected_samples']} (gate passed: {b['gate_passed']}). Test subset perplexity is {float(grouped['exp_mean_cross_entropy']):.3f}. Approximate Monte Carlo relative-perplexity interval: [{float(grouped['relative_perplexity_lower']):.4f}, {float(grouped['relative_perplexity_upper']):.4f}]. These five seeds quantify sampling variation of one frozen model on fixed text, not training or corpus uncertainty.\n",
        f"Selected free-node evaluations change from {a['test_free_node_updates']:,} to {b['test_free_node_updates']:,} ({100*(1-b['test_free_node_updates']/a['test_free_node_updates']):.2f}% reduction); sequential sampling depth is {a['sampling_depth']} versus {b['sampling_depth']}. Accuracy budgets are tolerance-matched, not numerically identical quality.\n",
        f"Range metadata rises from {a['range_bytes']:,} to {b['range_bytes']:,} bytes (0.0625 to 0.5 bits/weight). Field/range/retained-bias subtotals are {a['declared_field_range_bias_bytes']:,} and {b['declared_field_range_bias_bytes']:,} bytes. The new software observer additionally materializes {b['expanded_range_buffers_bytes_extra']:,} bytes of expanded lower/span arrays; these are explicitly excluded from that subtotal. Program objects, probability arrays, endpoint modes, stochastic state and accumulators cost more. This is not software compression or a measured density win.\n",
        'Each group has min/max endpoints represented by explicit clamping, counted as logical nodes but not free-node updates. More groups therefore move precision into range metadata and create more clamped endpoints; all effects are disclosed in grouped_block_resources.csv.\n',
        'The stronger conventional zero-preserving affine4 block128 control scores test perplexity 112.806 versus FP32 112.207. Useful native quality and possible sample reductions do not establish superiority over conventional quantization. CUDA accelerates software sampling; external arithmetic and expanded sampled values do not validate physical TSU MACs, latency or energy.\n',
        'Verification: exact same-path grouped operator tests on CPU/CUDA, all planned runs and raw operator lineage audited, plus independent pre/post-hook replay of every completed run. Raw directories: transformer_block0_group128 and transformer_block0_gpu. The long-horizon goal remains active for finite controls, nonuniform budgets, larger conversion and fresh evaluation data.\n'])
    (ROOT/'GROUPED_BLOCK_REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'selected':b,'test':grouped},indent=2))

if __name__=='__main__':main()
