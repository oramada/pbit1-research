"""Report every declared conventional control and preserve prior SPW outcomes."""
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path('results/thrml_spw/fine_range_quantization_controls')
def main():
    assert json.loads((OUT/'evidence_audit.json').read_text())['passed'] and json.loads((OUT/'comparison_audit.json').read_text())['passed'];p=json.loads((OUT/'protocol.json').read_text());m=json.loads((OUT/'manifest.json').read_text());data={r['condition']:r for r in m['scores']};pair=json.loads((OUT/'paired_comparisons.json').read_text());source=Path(p['source_cohort']);spw=json.loads((source/'manifest.json').read_text());samples={r['condition']:r for r in spw['summary']};fp=data['FP32'];int4=data['affine_zero_INT4_g16'];fine=samples['fine'];mean=data['programmed_mean_g16'];comparison=next(r for r in pair if r['native']=='fine' and r['control']=='affine_zero_INT4_g16');penalty=next(r for r in pair if r['native']=='fine' and r['control']=='programmed_mean_g16');nativebits={'fine':12.03125,'coarse':8.53125}
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained');colors={'minmax':'tab:blue','affine_zero':'tab:orange','symmetric':'tab:green'}
    for method,color in colors.items():
        for group,marker in [(16,'o'),(128,'x')]:
            rows=[r for r in m['scores'] if r['kind']=='uniform' and r['method']==method and r['group']==group];axes[0].scatter([r['bits_per_converted_weight_including_affine_bias'] for r in rows],[r['perplexity']/fp['perplexity'] for r in rows],color=color,marker=marker,label=f'{method}, group{group}',s=35)
    for name,r in samples.items():axes[0].scatter(nativebits[name],r['perplexity']/fp['perplexity'],marker='*',s=140,color='crimson',label=f'SPW {name}')
    axes[0].axhline(1,color='black',ls='--');axes[0].set(yscale='log',xlabel='Nominal code/range/bias bits per converted weight',ylabel='Perplexity / FP32 (log scale)',title='All 18 conventional lattices; consumed text');axes[0].legend(fontsize=6,ncol=2)
    names=['SPW fine T512','INT4 minmax g16','INT4 zero-preserving g16','INT8 minmax g16','Decoded SPW mean g16'];values=[fine['perplexity'],data['minmax_INT4_g16']['perplexity'],int4['perplexity'],data['minmax_INT8_g16']['perplexity'],mean['perplexity']];axes[1].barh(np.arange(5),100*(np.array(values)/fp['perplexity']-1));axes[1].set(yticks=np.arange(5),yticklabels=names,xlabel='Point perplexity change from FP32 (%)',title='Fixed references; original SPW strict gate fails');axes[1].invert_yaxis();fig.savefig('results/thrml_spw/fine_range_quantization_controls.png',dpi=180);plt.close(fig)
    text=f'''# Conventional controls for the single-p-bit neural result

On the same consumed text, zero-preserving INT4/group16 gives perplexity **{int4['perplexity']:.3f}** versus **{fine['perplexity']:.3f}** SPW/group16/T512 and **{fp['perplexity']:.3f}** FP32. Its nominal converted-weight code/range/bias payload is **{int4['bits_per_converted_weight_including_affine_bias']:.5f}** versus **12.03125 bits/weight** for SPW, excluding dynamic binary state and physical/runtime overhead. This ordinary numerical control is strong; a p-bit systems benefit requires separate hardware-cost evidence.

The paired diagnostic CE difference, SPW minus this INT4 reference, is **{comparison['CE_native_minus_control']:.9f}** nats/target with sensitivity interval **[{comparison['CE_sensitivity_low']:.9f}, {comparison['CE_sensitivity_high']:.9f}]**. The intervals use already-consumed text, five SPW sampling seeds and four context-bank clusters; deterministic controls are broadcast across seeds. They are unadjusted descriptive sensitivity comparisons, not fresh confirmation, familywise guarantees or a selected new method. All declared variants are reported below.

## Frozen controls and independent checks

All 21 conditions were frozen before any new diagnostic scoring: FP32; INT2/4/8 with minmax, zero-preserving affine and symmetric quantizers at groups16/128; and two deterministic decoders of the existing eight-bit p-bit controls. Scales/ranges use weights only, with no label fitting or post-hoc winner selection. Every method converts exactly the same 24 affine projections (42,467,328 weights, 51.84% of unique parameters) in the same pinned DistilGPT-2 checkpoint. Embeddings/head, norms and biases stay deterministic. Every condition uses the same 32 short teacher-forced test contexts as the completed native study. This is a consumed-text negative control, not another held-out experiment.

An independent implementation reconstructed all **891,813,888** weight coordinates, matched every quantized weight hash, reproduced every original-model token loss/prediction exactly, and checked all payload formulas. The 256-entry probability table was checked against 60-digit decimal arithmetic. A second independent auditor recomputed all **440,000** paired resampling values for 22 declared comparisons. No new p-bit samples, training or model selection occurred.

## Every deterministic condition

| Condition | PPL | Next-token accuracy (%) | Nominal bits/converted weight including affine bias | Mixed model parameter payload (MB) |
|---|---:|---:|---:|---:|
'''
    for variant in p['variants']:
        r=data[variant['name']];text+=f"| {variant['name']} | {r['perplexity']:.3f} | {100*r['token_accuracy']:.3f} | {r['bits_per_converted_weight_including_affine_bias']:.5f} | {r['mixed_parameter_payload_bytes']/1e6:.3f} |\n"
    text+=f'''
The mixed-model totals include all remaining FP32 parameters once. Biases already included in the converted affine payload are excluded from that remaining count. The supplementary native-study ledger's prior bias double-count (165,888 bytes per method) was corrected with the original ledger/source retained in `fine_range_confirmation/evidence_corrections/`; stochastic counts and every quality result are unchanged. Configuration bytes and the decoded-mean global lookup table are included in mixed totals. Algorithms/bit widths are declared global format descriptors. These are logical serialized descriptions, not measured simulator RAM or device area.

Minmax stores two F32 values per group; zero-preserving affine stores a F32 scale and a b-bit zero point; symmetric stores a F32 scale and leaves one signed code unused. Thus numeric code width alone is not the full payload. Group16 consists of consecutive input coefficients per output column, not16x16 tiles. All numerical inference here uses dequantized F32 tensors; no optimized packed integer kernel, hardware latency or energy has been measured.

## Encoded mean and finite sampling

The deterministic decoder of the same SPW/group16 codes gives PPL **{mean['perplexity']:.3f}**, close to FP32. It uses the same code/range payload plus a shared 2,048-byte F64 probability lookup table. The SPW-minus-decoded-mean CE difference is **{penalty['CE_native_minus_control']:.9f}**, interval **[{penalty['CE_sensitivity_low']:.9f}, {penalty['CE_sensitivity_high']:.9f}]** on this diagnostic. This is consistent with finite-sampling effects being important beyond the fixed control/range approximation. It is not an additive per-layer error decomposition or E[stochastic network]: an ordinary neural forward at programmed mean weights differs from averaging nonlinear stochastic networks. A deterministic decoder is a numerical reference, not a p-bit execution or physical-efficiency result.

The original native result remains unchanged: fine mean PPL+4.07%, next-token accuracy loss0.430pp; point rule passes, stricter uncertainty rule fails, and the original fine-versus-coarse CE interval includes zero. No control result rescues that failure or disproves Extropic. The representation has native simulator evidence; a systems advantage over efficient conventional representations remains unproved.

## Reproduction and scope

```bash
.venv/bin/python -m unittest research.thrml_spw.test_fine_range_quantization
OPENBLAS_NUM_THREADS=4 .venv/bin/python -m research.thrml_spw.verify_fine_range_quantization
.venv/bin/python -m research.thrml_spw.verify_quantization_comparisons
.venv-thrml/bin/python -m research.thrml_spw.fine_range_quantization_report
```

The immutable preparation and production modules refuse to overwrite completed conditions; protocol.json, raw token metrics, all weight hashes, codebook and audits are retained. These simple quantizers make no best-available GPTQ/AWQ claim. Longer-context generation, physical-device transfer, full parameter conversion, matched physical p-bit implementations and energy/latency/controller costs remain open. The broad goal remains active.
'''
    for old,new in [('groups16/128','groups 16/128'),('not16x16','not 16x16'),('PPL+4.07','PPL +4.07'),('loss0.430pp','loss 0.430 pp')]:text=text.replace(old,new)
    Path('results/thrml_spw/FINE_RANGE_QUANTIZATION_CONTROLS_REPORT.md').write_text(text);print(json.dumps({'controls_audited':21,'resampling_values_checked':440000,'INT4_affine_zero_group16_perplexity':int4['perplexity'],'SPW_fine_perplexity':fine['perplexity'],'new_pbit_samples':0,'goal_status':'active'},indent=2))
if __name__=='__main__':main()
