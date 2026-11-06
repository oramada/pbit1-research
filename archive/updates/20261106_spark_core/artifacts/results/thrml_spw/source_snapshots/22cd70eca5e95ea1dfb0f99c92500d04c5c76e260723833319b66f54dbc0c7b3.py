"""Known-parameter diagnostic bounds for the prior failed calibrated regime."""
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path('results/thrml_spw/transformer_calibration_floor');OLD=Path('results/thrml_spw/calibrated_transformer')


def read(name):
    with (OUT/name).open() as f:return list(csv.DictReader(f))


def main():
    audit=json.loads((OUT/'evidence_audit.json').read_text());assert audit['passed'];m=json.loads((OUT/'manifest.json').read_text());p=m['config']
    native=read('native_summary.csv');means=read('mean_summary.csv');contrasts=read('paired_diagnostics.csv');weight=json.loads((OUT/'weight_error_summary.json').read_text());components=weight['components_mean_squared']
    base=np.load(OLD/'baseline_tokens.npz');fp=float(np.exp(base['FP32_nll'].astype(float).mean()))
    labels=['Ideal\n(previous)','Measured\n(previous)','Known-parameter\noracle'];order=['ideal8','measured8','oracle8'];n=[next(r for r in native if r['condition']==k) for k in order]
    fig,ax=plt.subplots(2,2,figsize=(12,8),layout='constrained');xx=np.arange(3)
    v=[float(r['perplexity']) for r in n];low=[float(r['perplexity'])-np.exp(float(r['cross_entropy'])-float(r['CE_seed_normal95_margin'])) for r in n];high=[np.exp(float(r['cross_entropy'])+float(r['CE_seed_normal95_margin']))-float(r['perplexity']) for r in n]
    ax[0,0].errorbar(xx,v,yerr=[low,high],fmt='o',capsize=4);ax[0,0].axhline(fp,ls='--',color='black',label='Original FP32')
    ax[0,0].set(title='Native T1024; same profiles and MC keys',xticks=xx,xticklabels=labels,ylabel='Perplexity (lower is better)');ax[0,0].legend(fontsize=8)
    mo=['ideal8','measured8','oracle8','oracle_continuous_bound4'];g=[next(r for r in means if r['condition']==k) for k in mo]
    ax[0,1].plot(np.arange(4),[float(r['perplexity']) for r in g],'o');ax[0,1].axhline(fp,ls='--',color='black')
    ax[0,1].set(title='Analytic mean-weight diagnostic',xticks=np.arange(4),xticklabels=['Ideal8','Measured8','Oracle8','Oracle\ncontinuous B4'],ylabel='Perplexity; not E[noisy network]')
    keys=['measured_minus_known_SSE','oracle_grid_SSE','oracle_range_SSE'];ax[1,0].bar(np.arange(3),[components[k] for k in keys]);ax[1,0].axhline(components['total_SSE'],ls='--',color='black',label='Total measured-to-ideal MSE')
    ax[1,0].set_yscale('log');ax[1,0].set(title='Squared replacement effects; cross terms retained',xticks=np.arange(3),xticklabels=['Measured to\nknown params','Known grid\nversus continuous','Inverse-bound\nversus ideal'],ylabel='Mean-squared signed-weight effect');ax[1,0].legend(fontsize=7)
    y=np.array([float(r['CE_difference']) for r in contrasts]);lo=np.array([float(r['crossed_CE_low']) for r in contrasts]);hi=np.array([float(r['crossed_CE_high']) for r in contrasts]);ax[1,1].errorbar(np.arange(2),y,yerr=[y-lo,hi-y],fmt='o',capsize=4);ax[1,1].axhline(0,ls='--',color='black')
    ax[1,1].set(title='Paired crossed seed/bank sensitivity',xticks=np.arange(2),xticklabels=['Oracle minus\nmeasured','Oracle minus\nideal'],ylabel='CE difference (nats/target)')
    for a in ax.ravel():a.grid(axis='y',alpha=.2)
    fig.savefig('results/thrml_spw/transformer_calibration_floor_evidence.png',dpi=180);plt.close(fig)
    oracle=next(r for r in native if r['condition']=='oracle8');measured=next(r for r in native if r['condition']=='measured8');ideal=next(r for r in native if r['condition']=='ideal8')
    ledger={'new_native_forward_runs':5,'new_native_operator_runs':60,'new_native_probe_runs':0,'new_native_free_evaluations':m['native_free_evaluations'],'independent_native_replay_free_evaluations':m['native_free_evaluations'],
        'deterministic_mean_forward_runs':16,'old_measured_probe_evidence':'The unchanged oldP4096 actual counts are reused for analysis; no extra calibration observations are simulated. Oracle uses true parameters and has no implementable calibration-cost interpretation.',
        'free_weight_nodes':21233664,'exact_clamps':0,'nominal_controller_bits':8,'controller_bound':4,'native_T':1024,'contexts_per_bank':8,
        'scope':'All new native work is nondeployable known-parameter diagnostic plus independent verification. Mean forwards are conventional floating arithmetic. No zero-cost calibration, additional independent data, or physical energy/latency advantage claim.'}
    (OUT/'resource_ledger.json').write_text(json.dumps(ledger,indent=2)+'\n')
    text=['# Known-parameter controls separate calibration and controller-range effects\n',
        f"Native known-parameter controls score perplexity {float(oracle['perplexity']):.3f}, compared with {float(measured['perplexity']):.3f} for the old measured calibration and {float(ideal['perplexity']):.3f} for the old ideal-device case. Controller range[-4,4], eight-bit codes, T1024, device profiles, MC keys, text and converted weights remain fixed. The result points to the measured calibration estimates as the larger limitation in this tested regime. The oracle is a diagnostic that knows the true synthetic device parameters; it is not an implemented calibration solution.\n",
        '## Design and interpretation\n',
        'This is a mechanistic follow-up on the previous failed calibrated quarter-model study. It reuses the same already consumed64 WikiText-2 windows(4,096 targets), one pinned DistilGPT-2 checkpoint, five paired synthetic gain/offset profiles and the first three blocks\' twelve projections(25.92% of unique model parameters). Every normalized signed weight uses one free THRML p-bit with finite eight-bit controls; the other model parameters remain deterministic. This is not new held-out confirmation, full-model conversion or direct p=rawsignedweight.\n',
        'Four analytic programmed-field counterfactuals are fixed before scoring: ideal8 uses nominal controls; measured8 reuses the actual previous P4096 probe-derived commands; oracle8 uses the true gain/offset inverse, clips to[-4,4] and rounds to the same grid; oracle_continuous_bound4 uses the same bounded true inverse without grid rounding. True parameters enter only the explicit oracle diagnostics. The previous measured estimator still uses only its actual saved native counts, and every measured code is independently refit from those counts.\n',
        'Each physical field is emitted as F32 after F64 inverse/device arithmetic. Analytic marginals sigmoid(2field) are evaluated F64, mapped through the original shared lower/span ranges, then converted to F32 for ordinary mean-weight neural forwards. These sixteen deterministic forward runs are diagnostic; the mean-weight nonlinear network is not the expectation of the sampled network. Only one ideal mean-weight run is needed because it is identical across device profiles.\n',
        'Five new oracle8 native forward runs retain actual official THRML sampling, one free node per weight, T1024 accumulation before original nonlinearities and banks of eight contexts. Every later input is regenerated after preceding oracle stochastic outputs. True inverse fields and control rounding retain their physical-model imperfections; no program accesses the old native output arrays as replacements. No new calibration probes, quality gate, parameter search or outcome-based extension occurs.\n',
        '## Native task-quality diagnostics\n',
        '| Condition | Source | Perplexity | PPL / FP32 | Token accuracy (%) |\n|---|---|---:|---:|---:|'])
    for r in n:text.append(f"| {r['condition']} | {'new known-parameter diagnostic' if r['condition']=='oracle8' else 'previous audited run'} | {float(r['perplexity']):.3f} | {float(r['relative_perplexity_FP32']):.5f} | {100*float(r['token_accuracy']):.3f} |")
    text.extend([f"\nOriginal FP32 perplexity is {fp:.3f}. The old measured result remains a failed5% quality regime. The oracle result cannot upgrade it to a successful deployable method, even where its numerical quality is close to the ideal case.\n",
        '| Paired contrast | CE difference | PPL ratio | Crossed CE sensitivity interval | Accuracy delta (pp) | Crossed accuracy interval (pp) |\n|---|---:|---:|---|---:|---|'])
    for r in contrasts:text.append(f"| {r['contrast']} | {float(r['CE_difference']):+.6f} | {float(r['perplexity_ratio']):.6f} | [{float(r['crossed_CE_low']):+.6f}, {float(r['crossed_CE_high']):+.6f}] | {100*float(r['accuracy_difference']):+.3f} | [{100*float(r['crossed_accuracy_low']):+.3f}, {100*float(r['crossed_accuracy_high']):+.3f}] |")
    text.extend(['\nIntervals inherit the source study\'s descriptive crossed resampling convention: five joint device/sampling seeds crossed with eight whole8-context banks,10,000 resamples and fixed seed2026114701. They are sensitivity intervals under exchangeable clusters, not formal coverage or equivalence. Oracle-versus-ideal CE uncertainty includes zero; that is not an equivalence test. The paired oracle-versus-measured accuracy interval also includes zero, even though its CE contrast is clearly negative in this diagnostic. Text and pretrained model are reused throughout.\n',
        '## Analytic mean-weight diagnostics\n',
        '| Programmed fields | Device profiles | Mean-weight perplexity | Token accuracy (%) |\n|---|---:|---:|---:|'])
    for r in means:text.append(f"| {r['condition']} | {r['device_profiles']} | {float(r['perplexity']):.3f} | {100*float(r['token_accuracy']):.3f} |")
    text.extend(['\nPPL is exp(mean CE), rather than the arithmetic mean of per-profile perplexities. Continuous-control and true-parameter diagnostics are not physical controller demonstrations. Differences among these means cannot be added to native task loss as independent noise components.\n',
        '## Exact weight-error telescope\n',
        'For the same coordinate/profile, define e_est=W_measured-W_oracle8, e_grid=W_oracle8-W_oracle_continuous_bound4 and e_range=W_oracle_continuous_bound4-W_ideal. Their sum is exactly W_measured-W_ideal. Squared error includes all signed cross terms; it is not an additive independent-noise attribution. The estimator replacement effect includes finite-data estimator choices and their interaction with rounding. It does not isolate unbiased calibration variance alone.\n',
        '| Mean-squared term across all coordinate/profile pairs | Value |\n|---|---:|'])
    for k,v in components.items():text.append(f'| {k} | {v:.10g} |')
    text.extend([f"\nThese totals cover {weight['weights_across_device_profiles']:,} weight/profile coordinates. The known inverse exceeds the controller bound for {100*weight['known_inverse_outside_bound_fraction']:.4f}% of coordinates. That fraction does not determine its task impact: bounded targets can already be nearly saturated. The squared measured-to-known replacement effect is {components['measured_minus_known_SSE']/components['oracle_range_SSE']:.1f} times the squared range effect, with nonzero cross terms retained. This is a diagnostic comparison, not a percentage of task loss explained.\n",
        '## Independent evidence and cost\n',
        'An independent program regenerates every field and F32 mean matrix, refits all old measured commands from immutable native counts, checks all60 weight-error decompositions including signed cross terms, and replays every token loss/prediction in all16 ordinary mean-weight forwards. Independently constructed public THRML frequency programs replay every node in all60 new native oracle operators; a separate original-model pre/post-hook pass checks every input and final token loss/prediction in all five complete native cases. Paired task scores, all40,000 diagnostic bootstrap values and resource counts are independently recomputed. Source/data/control hashes and all old negative outcomes are retained.\n',
        f"New native inference executes {m['native_free_evaluations']:,} observed free-node updates; independent native replay repeats that count for verification. There are zero new native probe runs, because measured counts come from the old study and the oracle is granted true parameters. This zero is not an implementable calibration cost. The new diagnostics do not change the earlier control payload8.53125bits/converted weight or establish physical energy/latency or memory advantages.\n",
        'The broad goal remains active. A practical follow-up should test probe placement, budget or estimator changes with matched resources and prospective validation-only choices, followed by separate confirmation. The oracle suggests such work is more promising than simply widening the tested inverse-control range, but does not prove any particular new estimator or budget will meet quality targets. Physical device transfer laws, drift/correlation, full-model conversion and system costs remain open.\n',
        '## Reproduction\n',
        'The wrapper resumes the frozen driver only when the native manifest is absent; for completed studies it verifies hashes without changing elapsed-time metadata or sampling again. All paths are relative to repository root.\n',
        '```bash\nOPENBLAS_NUM_THREADS=8 .venv/bin/python -m research.thrml_spw.reproduce_transformer_floor\nOPENBLAS_NUM_THREADS=4 .venv/bin/python -m research.thrml_spw.audit_transformer_floor_means\nOPENBLAS_NUM_THREADS=4 .venv/bin/python -m research.thrml_spw.gpu_environment -- .venv-thrml-gpu/bin/python -m research.thrml_spw.audit_transformer_floor_native\nOPENBLAS_NUM_THREADS=4 .venv/bin/python -m research.thrml_spw.sequential_hook_audit --root results/thrml_spw/transformer_calibration_floor --tokens results/thrml_spw/calibrated_transformer/tokens.npz\n.venv/bin/python -m research.thrml_spw.transformer_floor_statistics\n.venv/bin/python -m research.thrml_spw.verify_transformer_floor\n.venv-thrml/bin/python -m research.thrml_spw.transformer_floor_report\n```\n'])
    rendered='\n'.join(text)+'\n'
    replacements={'range[-4':'range [-4','consumed64':'consumed 64','windows(':'windows (','projections(':'projections (','rawsignedweight':'raw signed weight','to[-4':'to [-4','failed5%':'failed 5%','whole8':'whole 8','banks,10,000':'banks, 10,000','seed2026114701':'seed 2026114701','all60':'all 60','all16':'all 16','all40,000':'all 40,000','payload8.53125bits':'payload 8.53125 bits'}
    for a,b in replacements.items():rendered=rendered.replace(a,b)
    Path('results/thrml_spw/TRANSFORMER_CALIBRATION_FLOOR_REPORT.md').write_text(rendered)
    print(json.dumps({'native':native,'means':means,'contrasts':contrasts,'weight_errors':weight,'resource_ledger':ledger},indent=2))


if __name__=='__main__':main()
