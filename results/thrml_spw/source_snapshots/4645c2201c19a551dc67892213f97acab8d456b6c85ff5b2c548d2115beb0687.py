"""Render bounded native operator evidence; do not infer neural quality."""
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path('results/thrml_spw/range_localization')
def main():
    a=json.loads((OUT/'evidence_audit.json').read_text());assert a['completed']
    with (OUT/'probe_summary.csv').open() as f: rows=list(csv.DictReader(f))
    with (OUT/'mean_model_metrics.csv').open() as f: means=list(csv.DictReader(f))
    contrasts=json.loads((OUT/'paired_operator_ratios.json').read_text());primary=contrasts[0]
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for condition in ['g128_T1024','g128_T512','g16_T512']:
        r=[v for v in rows if v['condition']==condition]
        axes[0].plot([int(v['stage']) for v in r],[float(v['observed_over_prediction']) for v in r],'.-',label=condition)
    axes[0].axhspan(.8,1.2,alpha=.12);axes[0].axhline(1,color='black',ls='--');axes[0].set(xlabel='Projection',ylabel='Observed / predicted target MSE');axes[0].legend(fontsize=8)
    for i,r in enumerate(contrasts):
        y=r['geometric_MSE_ratio'];axes[1].errorbar(i,y,yerr=[[y-r['approx_seed_normal95_low']],[r['approx_seed_normal95_high']-y]],fmt='o',capsize=5)
    axes[1].axhline(1,color='black',ls='--');axes[1].set(xticks=[0,1,2],xticklabels=['Fine512 / coarse1024','Fine512 / coarse512','Coarse512 / coarse1024'],ylabel='Geometric target MSE ratio');axes[1].tick_params(axis='x',labelsize=7)
    fig.savefig('results/thrml_spw/range_localization_evidence.png',dpi=180);plt.close(fig)
    text=f'''# Local range encoding: native operator evidence

Raw native group-16 ranges at T512 give geometric operator MSE ratio **{primary['geometric_MSE_ratio']:.6f}** relative to group-128 at T1024 (approximate five-seed interval **[{primary['approx_seed_normal95_low']:.6f}, {primary['approx_seed_normal95_high']:.6f}]**). This is error against original deterministic linear operators on fixed validation inputs. It is not a full stochastic neural task-quality result.

The cost is more control/range payload: **12.03125 versus 8.53125 bits per converted weight**, plus 48 group-configuration bytes for group16 (group128 uses the established implicit configuration). Both designs use one free p-bit per normalized weight and eight-bit bias controls. The fine design halves observed node updates. This supports an operator precision/metadata tradeoff subject to the unresolved exact-output replay gate; it establishes no physical energy or latency advantage.

The frozen protocol covers all 24 affine projections in six blocks of one pinned DistilGPT-2 checkpoint: 42,467,328 converted weights, 51.84% of unique model parameters. Embeddings/head, norms and biases stay deterministic. Two fixed teacher-context positions from each of 32 already-consumed validation windows give 64 input rows per projection. One sampled weight bank is shared across those rows. There are three fixed conditions and five independent Monte Carlo seeds, giving 360 full-matrix native operators. No new held-out test text was scored.

Official THRML 0.1.4 at commit 2d54cd8404df85bb65703d7e8b4e84e48c1b72a9 samples independent unary Gibbs p-bits with finite fields [-4,4], quantized to eight bits. All converted nodes remain free. Native workers receive controls and input arrays; original weights are used only in separately preserved error references. Stochastic contributions are accumulated before subsequent nonlinearities.

## Mathematical and independent checks

For a weight inside its group interval, ideal one-draw variance is (w-l)(u-w). Nested intervals therefore reduce variance coordinatewise; all 127,401,984 comparisons across group sizes 16/32/64/128 were checked against original checkpoint weights. Quantized finite controls can violate this coordinatewise ordering, so their actual probability variance and mean bias are computed separately. Target MSE prediction includes sampling variance divided by T and programmed-mean bias squared.

The independent public-factor/SumOnly byte-exact output gate passed: **{a["native_exact_output_gate_passed"]}**. Exact failures remain recorded; even a fresh invocation of the production implementation differed slightly from preserved outputs in a diagnostic. Compiler/runtime arithmetic reproducibility remains unresolved. A separate frequency observer recovers integer counts; all frequency-based references satisfy the derived IEEE floating-point envelope. Separate observers preserve the actual compiled F32 execution. Original neural anchor inputs, deterministic targets, controls, raw scores and resource counters have independent checks. The fixed practical theory gate (all 72 layer/condition means within 20% of prediction) is **{a['all_predeclared_theory_checks_within20percent']}**; observed/predicted ratios range from {a['min_observed_over_prediction']:.6f} to {a['max_observed_over_prediction']:.6f}. This gate is not a formal coverage statement.

Total executed native node updates, including independent output/frequency replays, failed replay prefixes and diagnostic calls: **{a['total_executed_native_updates']:,}**. Five-seed intervals describe Monte Carlo sensitivity on these fixed inputs and layers; they do not establish architecture generalization.

## Deterministic programmed-mean diagnostic

The following ordinary deterministic neural forwards replace converted matrices by their programmed means. They are **not** expected stochastic-network outputs or full stochastic neural confirmations.

'''
    text+='| Condition | Perplexity |\n|---|---:|\n'
    for r in means:text+=f"| {r['condition']} | {float(r['perplexity']):.6f} |\n"
    text+='''
Group64 mean-model perplexity is worse than group128 despite smaller ideal variances. Lower operator variance alone does not guarantee better neural quality.

## Remaining confirmation and reproduction

Resolve the failed exact-output replay gate before claiming complete native reproducibility. A prospective full stochastic neural comparison on unused test text remains required. The persistent research volume has less than 1 GB available and no second persistent volume is mounted; at least 10 GB or another persistent path has been requested. No volatile-only evidence, changed quality thresholds, or operator surrogate is substituted for that confirmation. The long-horizon research goal remains active. These results do not disprove Extropic or establish hardware superiority.

Artifacts: protocol.json; preparation_audit.json; analytic_evidence_audit.json; native_evidence_audit.json; evidence_audit.json; analytic_operator_metrics.csv; probe_summary.csv; paired_operator_ratios.json; original anchors; all 360 output arrays and native manifests. Controls and source hashes are recorded before execution.

```bash
OPENBLAS_NUM_THREADS=4 .venv/bin/python -m research.thrml_spw.range_localization_analytic
OPENBLAS_NUM_THREADS=4 .venv/bin/python -m research.thrml_spw.verify_range_localization_analytic
OPENBLAS_NUM_THREADS=4 .venv/bin/python -m research.thrml_spw.range_localization_probes
OPENBLAS_NUM_THREADS=4 .venv/bin/python -m research.thrml_spw.gpu_environment -- .venv-thrml-gpu/bin/python -m research.thrml_spw.verify_range_localization_native
.venv/bin/python -m research.thrml_spw.range_localization_statistics
.venv-thrml/bin/python -m research.thrml_spw.range_localization_report
```

Completed analytic/native drivers validate caches without drawing new samples; independent replay deliberately repeats preserved keys. Preparation is a one-time frozen operation.
'''
    Path('results/thrml_spw/RANGE_LOCALIZATION_REPORT.md').write_text(text)
    print(json.dumps({'primary':primary,'theory_gate':a['all_predeclared_theory_checks_within20percent'],'goal_status':'active'},indent=2))
if __name__=='__main__':main()
