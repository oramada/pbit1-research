"""Render the audited frozen neural comparison without selecting new settings."""
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .fine_range_common import OUT
def main():
    audit=json.loads((OUT/'evidence_audit.json').read_text());assert audit['passed'];p=json.loads((OUT/'protocol.json').read_text());m=json.loads((OUT/'manifest.json').read_text());pair=json.loads((OUT/'paired_summary.json').read_text())
    with (OUT/'summary.csv').open() as f:summary={r['condition']:r for r in csv.DictReader(f)}
    with (OUT/'baseline_metrics.csv').open() as f:baseline=next(csv.DictReader(f))
    fine,coarse=summary['fine'],summary['coarse'];interpretation='includes zero' if pair['CE_sensitivity_low']<=0<=pair['CE_sensitivity_high'] else ('is below zero' if pair['CE_sensitivity_high']<0 else 'is above zero');payload={str(g):{'bytes':sum(r['parameter_payload_bytes'] for r in controls),'configuration_bytes':sum(r['group_flag_bytes'] for r in controls),'bits_per_converted_weight':8*sum(r['parameter_payload_bytes'] for r in controls)/p['converted_weights']} for g,controls in [(int(k),v) for k,v in p['control_artifacts'].items()]}
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained');xx=np.arange(2);values=[coarse,fine]
    axes[0].plot(xx,[100*(float(r['relative_perplexity'])-1) for r in values],'o',label='Point');axes[0].plot(xx,[100*(float(r['bootstrap_relative_upper'])-1) for r in values],'_',ms=14,label='Upper sensitivity bound');axes[0].axhline(5,color='black',ls='--',label='Frozen 5% rule');axes[0].set(xticks=xx,xticklabels=['Group128 / T1024','Group16 / T512'],ylabel='Perplexity increase over FP32 (%)');axes[0].legend(fontsize=8)
    axes[1].bar([-.18,.82],[1,1],width=.36,label='Coarse');axes[1].bar([.18,1.18],[.5,payload['16']['bits_per_converted_weight']/payload['128']['bits_per_converted_weight']],width=.36,label='Fine');axes[1].set(xticks=[0,1],xticklabels=['Observed node updates','Control/range payload'],ylabel='Ratio to coarse');axes[1].legend(fontsize=8);fig.savefig('results/thrml_spw/fine_range_confirmation_evidence.png',dpi=180);plt.close(fig)
    text=f'''# Finer-range stochastic neural confirmation

The frozen finer-range method gives perplexity **{float(fine['perplexity']):.3f}** versus **{float(coarse['perplexity']):.3f}** for the coarse comparator and **{float(baseline['perplexity']):.3f}** FP32. Fine point quality gate: **{fine['point_passed']}**; strict gate: **{fine['passes']}**. Coarse point gate: **{coarse['point_passed']}**; strict gate: **{coarse['passes']}**. The paired CE sensitivity interval {interpretation}: [{pair['CE_sensitivity_low']:.9f}, {pair['CE_sensitivity_high']:.9f}] nats/target, fine minus coarse. These outcomes are retained under the original thresholds; neither audit correctness nor earlier operator MSE predicts a quality-gate success.

## Prospective setup and actual neural path

The candidate was fixed from consumed-validation operator evidence before any scoring on 32 unused test contexts (65 tokens each, 2,048 targets). Fine uses group16/T512 and coarse group128/T1024, both finite eight-bit [-4,4] unary controls, five paired seeds and four eight-context banks. Each layer's native linear sum is averaged before subsequent nonlinearities. Every later projection receives the current stochastic student's actual activations, not FP32 teacher inputs. Sampling keys root1210730 fold seed/stage/bank; different T is not a shared sample prefix. No candidate is selected or rescued from this test.

All 24 affine projections in six DistilGPT-2 blocks are converted: 42,467,328 normalized signed weights, 51.84% of unique parameters. Every converted weight uses one free logical p-bit. Embeddings/head, norms and biases remain deterministic. Official THRML 0.1.4 at commit2d54cd8404df85bb65703d7e8b4e84e48c1b72a9 and fixed CUDA/JAX F32/highest arithmetic are used, with XLA timing-based autotuning disabled. This is a full stochastic student forward through the converted matrices, not conversion of every model parameter.

The independent original-model audit reproduces FP32 losses/predictions, every actual student-input NPY hash and all final native-model token losses/predictions. A separate public-factor/SumOnly native replay checks all240 operators byte for byte, with separate integer counts/F64 references and derived IEEE bounds. Descriptive 2e-5 flags remain visible. The independent statistical auditor recomputes all60,000 paired seed/bank resampling values, all raw score rows, fixed quality rules and resource counters.

## Resource tradeoff and limits

Fine halves observed node updates and serialized sampling rounds. Control/range payload rises from **{payload['128']['bits_per_converted_weight']:.5f}** to **{payload['16']['bits_per_converted_weight']:.5f} bits per converted weight**, plus **{payload['16']['configuration_bytes']}** configuration bytes for group16. Those payloads include finite codes, group lower/span values and deterministic biases; runtime buffers and physical controller/readout/accumulator costs are additional. One logical p-bit per weight does not mean one total stored bit. No physical energy, latency or system-density advantage is inferred.

Production native node updates: **{m['native_observed_updates']:,}**; independent replay costs are reported separately in native_audit.json. Static independent ideal devices, one checkpoint, short contexts and five seeds bound this result. The crossed bootstrap intervals describe seed/bank sensitivity, not guaranteed population coverage. Measured/correlated/drifting devices, longer contexts, other models, probability-aware training and full parameter conversion remain separate work. The broad research goal remains active, and no Extropic architecture is disproved.

Historical evidence was losslessly archived with original bytes and hashes preserved to obtain execution capacity. See [archive restoration instructions](lossless_archive/README.md) before rerunning older commands that require archived original paths. This storage representation change is unrelated to model compression.

## Launcher recovery and provenance

All ten model runs and 240 native operators completed before the launcher exceeded its 60-second worker-shutdown wait. The original failure log and frozen production sources are retained in execution_incidents/. A separate summary-only finalizer verified the completed records and rebuilt the manifest and statistics with **zero new model runs or p-bit samples**. The manifest records this incident and leaves total launcher elapsed time unspecified; individual operation timings and the last native-stage log time remain available. Independent audits verify the saved inference evidence separately from that launcher failure.

## Reproduction

```bash
OPENBLAS_NUM_THREADS=4 .venv/bin/python -m research.thrml_spw.fine_range_confirmation
OPENBLAS_NUM_THREADS=4 .venv/bin/python -m research.thrml_spw.audit_fine_range_paths
.venv/bin/python -m research.thrml_spw.verify_fine_range_statistics
.venv-thrml/bin/python -m research.thrml_spw.fine_range_report
```

The frozen execution/source manifests, original token reservation and all native outputs are retained. The production driver checks available capacity even when invoked again; completed output cases are checked by hashes. If a fresh invocation completes all ten cases but fails only during shutdown, `python -m research.thrml_spw.finalize_fine_range` performs hash-checked summary-only recovery; it refuses to overwrite a manifest. Existing data can be independently audited without rerunning production. Independent audits deliberately repeat preserved keys. Original earlier quality and numerical failures are unchanged.
'''
    for old,new in [('root1210730','root 1210730'),('commit2d','commit 2d'),('all240','all 240'),('all60,000','all 60,000')]:text=text.replace(old,new)
    Path('results/thrml_spw/FINE_RANGE_CONFIRMATION_REPORT.md').write_text(text);print(json.dumps({'summary':summary,'paired':pair,'payload':payload,'goal_status':'active'},indent=2))
if __name__=='__main__':main()
