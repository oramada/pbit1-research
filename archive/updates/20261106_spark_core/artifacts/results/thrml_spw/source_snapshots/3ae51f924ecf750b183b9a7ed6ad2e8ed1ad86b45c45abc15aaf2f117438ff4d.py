"""Audited full-network drift quality and explicit scheduled-refresh costs."""
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path('results/thrml_spw/clocked_mnist')
POLICIES=['retain_initial','refresh_every32_batches']


def read(path):
    with path.open() as f:return list(csv.DictReader(f))


def margin(x):return float(1.96*np.std(x,ddof=1)/np.sqrt(len(x)))


def main():
    audit=json.loads((OUT/'evidence_audit.json').read_text());replay=json.loads((OUT/'public_path_audit.json').read_text());assert audit['passed'] and replay['passed']
    native=read(OUT/'native_metrics.csv');windows=read(OUT/'window_metrics.csv');base=read(OUT/'baseline_metrics.csv');result=json.loads((OUT/'REPORT.json').read_text())
    fp=np.array([float(next(r['accuracy'] for r in base if int(r['model'])==m)) for m in range(5)])
    vectors={};summary=[];contrasts=[];window_summary=[]
    for rate in [0.,.01,.04]:
        for policy in POLICIES:
            a=np.array([np.mean([float(r['accuracy']) for r in native if int(r['model'])==m and float(r['rate'])==rate and r['policy']==policy]) for m in range(5)])
            loss=np.array([np.mean([float(r['cross_entropy']) for r in native if int(r['model'])==m and float(r['rate'])==rate and r['policy']==policy]) for m in range(5)])
            vectors[rate,policy]=a;delta=a-fp
            summary.append({'rate_SD_per1024_sweeps':rate,'policy':policy,'trained_models':5,'joint_realizations_per_model':5,'accuracy':float(a.mean()),'model_normal95_margin':margin(a),'paired_FP32_difference':float(delta.mean()),'paired_model_normal95_margin':margin(delta),'cross_entropy':float(loss.mean())})
            for window in range(10):
                a=np.array([np.mean([float(r['accuracy']) for r in windows if int(r['model'])==m and float(r['rate'])==rate and r['policy']==policy and int(r['window'])==window]) for m in range(5)])
                window_summary.append({'rate':rate,'policy':policy,'window':window,'examples':1024 if window<9 else 784,'accuracy':float(a.mean()),'model_normal95_margin':margin(a)})
        delta=vectors[rate,POLICIES[1]]-vectors[rate,POLICIES[0]]
        contrasts.append({'rate':rate,'contrast':'refresh-minus-retain','paired_accuracy_difference':float(delta.mean()),'paired_model_normal95_margin':margin(delta)})
    for name,rr in [('summary.csv',summary),('paired_contrasts.csv',contrasts),('window_summary.csv',window_summary)]:
        with (OUT/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
    # Diagnostic only: true device/path values quantify error in the measured fit,
    # never re-enter its commands or inference. End-of-probe field is query onset.
    diagnostics=[]
    for model in range(5):
        for device in range(5):
            noise=np.random.default_rng(2026110500+100*model+device).standard_normal((10656,2368));path=np.concatenate([np.zeros((1,2368)),np.cumsum(noise,axis=0)],axis=0)/4;del noise
            for rate in [0.,.01,.04]:
                a=np.load(OUT/'calibration'/f'model{model}_device{device}_rate{rate:g}.npz')
                nominal=(-4+a['nominal_codes'].astype(float)*8/255).astype(np.float32).astype(float)
                for window in range(10):
                    gain,offset=a['gain_estimates'][window],a['offset_estimates'][window]
                    requested=(nominal-offset)/gain;truth=a['initial_offset']+rate*path[window*1088+64]
                    diagnostics.append({'model':model,'device':device,'rate':rate,'window':window,
                        'gain_fit_rmse':float(np.sqrt(np.mean((gain-a['gain'])**2))),
                        'offset_fit_rmse_at_query_start':float(np.sqrt(np.mean((offset-truth)**2))),
                        'requested_commands_clipped':int(np.count_nonzero(np.abs(requested)>4)),
                        'emitted_codes_at_rails':int(np.count_nonzero((a['corrected_codes'][window]==0)|(a['corrected_codes'][window]==255)))})
            del path
    with (OUT/'fit_diagnostics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(diagnostics[0]));w.writeheader();w.writerows(diagnostics)
    infer=313*1024*2368;ledger=[]
    for policy in POLICIES:
        refresh=policy==POLICIES[1];cal=4096*(10 if refresh else 1);idle=0 if refresh else 36864
        ledger.append({'policy':policy,'weight_pbits':2368,'free_pbits':2368,'exact_clamps':0,'neural_bias_values':42,
            'command_codes_uint8_bytes':2368,'calibration_count_buffer_if_int32_bytes':2*2368*4,
            'inference_free_evaluations':infer,'calibration_free_evaluations':cal*2368,
            'active_calibration_sweeps':cal,'active_serial_layer_sweeps':641024,'idle_sweeps':idle,'matched_elapsed_sweep_clock':681984,
            'active_free_evaluation_multiplier_vs_retain':(infer+cal*2368)/(infer+4096*2368),
            'active_serial_sweep_multiplier_vs_retain':(641024+cal)/(641024+4096),
            'scope':'One complete10000-image trajectory with batches32 and16padding; native observed updates only. Shared comparison clock inserts idle into retain. No idle/autonomous switching, fitting/programming/readout energy or real time modeled.'})
    (OUT/'resource_ledger.json').write_text(json.dumps({'candidate_trajectories':ledger,'production_inference_free_evaluations':result['production_inference_free_evaluations'],'production_calibration_free_evaluations':result['production_calibration_free_evaluations'],
        'verification_inference_free_evaluations':result['production_inference_free_evaluations'],'verification_calibration_free_evaluations':result['production_calibration_free_evaluations'],
        'note':'Production calibration shared initial/refresh events across policies; candidate costs do not double-charge shared evidence. Replay counts verification work, not additional independent statistical realizations. Command/count buffer subtotal excludes runtime/state/field/history arrays.'},indent=2)+'\n')
    fig,ax=plt.subplots(2,2,figsize=(12,8),layout='constrained');xx=np.arange(3)
    for pi,policy in enumerate(POLICIES):
        g=[r for r in summary if r['policy']==policy];ax[0,0].errorbar(xx,[100*r['accuracy'] for r in g],yerr=[100*r['model_normal95_margin'] for r in g],fmt='o-',label=policy.replace('_',' '))
    ax[0,0].axhline(fp.mean()*100,ls='--',color='black',label='FP32');ax[0,0].set(title='Complete 10,000-image serial trajectory',xticks=xx,xticklabels=['0','0.01','0.04'],xlabel='Drift SD per 1,024 sweeps',ylabel='Accuracy (%)');ax[0,0].legend(fontsize=8)
    for axis,rate in [(ax[0,1],.01),(ax[1,0],.04)]:
        for policy in POLICIES:
            g=[r for r in window_summary if r['rate']==rate and r['policy']==policy]
            axis.errorbar(np.arange(1,11),[100*r['accuracy'] for r in g],yerr=[100*r['model_normal95_margin'] for r in g],fmt='o-',label=policy.replace('_',' '))
        axis.axhline(fp.mean()*100,ls='--',color='black');axis.set(title=f'Chronological windows: drift SD {rate}',xlabel='Query window (32 batches; last 25)',ylabel='Accuracy (%)');axis.legend(fontsize=8)
    a=ax[1,1];a.bar([0,1],[r['inference_free_evaluations']/1e6 for r in ledger],label='Layer sampling')
    a.bar([0,1],[r['calibration_free_evaluations']/1e6 for r in ledger],bottom=[r['inference_free_evaluations']/1e6 for r in ledger],label='Observed probes')
    a.set(title='Active updates; both clocks 681,984 sweeps',xticks=[0,1],xticklabels=['Retain + idle','Refresh'],ylabel='Million observed free-node evaluations');a.legend(fontsize=8)
    for a in ax.ravel():a.grid(axis='y',alpha=.2)
    fig.savefig('results/thrml_spw/clocked_mnist_evidence.png',dpi=180);plt.close(fig)
    primary=contrasts[2];text=['# Full-network drift with scheduled native calibration\n',
        f"All-free finite-endpoint networks retain {100*vectors[.01,POLICIES[1]].mean():.3f}% accuracy with scheduled refresh at the slower synthetic drift rate, versus {100*vectors[.01,POLICIES[0]].mean():.3f}% retaining the initial calibration. At the faster rate, refresh scores {100*vectors[.04,POLICIES[1]].mean():.3f}% versus {100*vectors[.04,POLICIES[0]].mean():.3f}%; recovery remains well below {100*fp.mean():.3f}% FP32. The primary fast-rate paired improvement is {100*primary['paired_accuracy_difference']:.3f} percentage points with approximate five-model normal margin {100*primary['paired_model_normal95_margin']:.3f}.\n",
        '## Fixed experiment and scope\n',
        'Every one of 2,368 learned matrix weights is a free THRML SpinNode, with no endpoint clamps or affine normalization. Finite eight-bit fields in [-4,4] approximate endpoint targets using epsilon=sigmoid(-8), rather than exactly realizing zero/one. The 42 neural biases stay deterministic. Five existing positive-weight 64->32->10 models and five independently generated joint device/path/sampling realizations per model each evaluate the complete 10,000-image existing MNIST test set. No retraining or test-driven selection occurs. A fixed shuffled input order avoids assigning original dataset order to device age; this does not make the previously used test set fresh.\n',
        'All 150 predeclared full-test cases execute at T1024 per layer with 64-sweep piecewise fields. Positive static device gain is lognormal mean1/log-SD0.2; initial offset SD0.2. Offset then follows independent Brownian increments on every device, normalized to SD0,0.01,0.04 per1024 native sweeps. No physical drift-rate calibration exists. Independent unary Gibbs conditionals are the transition model: no temporal/cross-device persistence or readout noise is added.\n',
        '## Absolute clock and estimator\n',
        'Each of313 batches shares one sampled weight state per layer draw across32 examples. Layer0 runs1024 sweeps and accumulates linear outputs before ReLU; layer1 then runs1024 sweeps. Fields change every64 sweeps throughout both layers. Both weight banks continue to drift on the absolute clock even while a bank is not sampled. Sixteen padded examples are discarded. Segment and dataset vmap is software evaluation of fixed serial-clock histories, not claimed hardware parallelism. Segment initial states resetfalse; independent unary conditionals already resample their marginal on the first actual sweep, so no extra burn-in or autocorrelation benefit is assumed.\n',
        'A4096-sweep calibration window precedes batches0,32,...288. Each contains2048 samples at a finite code near field-1 followed by2048 near+1; bias drifts every64 sweeps inside both probes. A Jeffreys-smoothed-logit fit estimates one gain/offset line, clips gain to the predeclared[0.25,4] guard, and emits finite inverse commands. This fit sees only measured counts and commanded probes. It never receives true device parameters, drift values or task labels.\n',
        'Retain-initial uses the first measured codes and idles in the nine later calibration windows; refresh-every32 uses each new fit. Thus input times and the entire latent Brownian path match between policies. Both trajectories finish at681984 sweeps:641024 layer sweeps plus40960 calibration/idle sweeps. Retain spends4096 on active probes and36864 idle; refresh spends40960 on active probes. This is a finite predeclared periodic schedule over one test stream, not an optimized or steady-state refresh guarantee.\n',
        '## All task results\n',
        '| Drift SD /1024 sweeps | Policy | Accuracy (%) | Five-model normal margin (pp) | Paired FP32 difference (pp) | Paired margin (pp) |\n|---:|---|---:|---:|---:|---:|']
    for r in summary:text.append(f"| {r['rate_SD_per1024_sweeps']:g} | {r['policy']} | {100*r['accuracy']:.3f} | {100*r['model_normal95_margin']:.3f} | {100*r['paired_FP32_difference']:+.3f} | {100*r['paired_model_normal95_margin']:.3f} |")
    text.extend(['\n| Drift SD | Refresh minus retain (pp) | Paired model normal margin (pp) |\n|---:|---:|---:|'])
    for r in contrasts:text.append(f"| {r['rate']:g} | {100*r['paired_accuracy_difference']:+.3f} | {100*r['paired_model_normal95_margin']:.3f} |")
    text.extend(['\nFive joint device/path/sampling realizations are averaged within each of five trained models first; the displayed margin is1.96*SD/sqrt(5) over those model means. Paired contrasts use model-level paired differences. These small-sample normal margins are descriptive, not formal coverage/equivalence guarantees. All chronological windows, including the last784-example window, are retained. Window-level class composition differs; they are not independent new training seeds.\n',
        '## Cost and remaining failures\n',
        f"One trajectory uses {infer:,} observed inference free-node updates. Retain uses {ledger[0]['calibration_free_evaluations']:,} probe updates and refresh {ledger[1]['calibration_free_evaluations']:,}. Refresh therefore needs {100*(ledger[1]['active_free_evaluation_multiplier_vs_retain']-1):.3f}% more active observed node updates, and {100*(ledger[1]['active_serial_sweep_multiplier_vs_retain']-1):.3f}% more active serial sweeps if comparison-only retain idle is removed. Equal comparison clocks do not demonstrate equal deployment latency or energy.\n",
        'Per-device commands require2368 uint8 bytes, plus measured-count fitting buffers and neural biases; they are not one total storage bit per weight. Autonomous unobserved p-bit switching, readout/accumulation energy, real-time drift between physical readouts, fitting/programming time and physical bias retention are outside this model. Active observed Gibbs counts cannot be interpreted as total hardware switching energy.\n',
        'At the faster rate, periodic refresh still fails to approach the static FP32 quality. Probe drift, within-query drift and controller limits are all present; this experiment does not causally isolate their contributions. True-parameter diagnostics quantify measured fit errors at query onset and finite-rail commands separately, without feeding them to the estimator. No hardware-compatible optimum refresh rate or universal success is established.\n',
        '## Independent evidence\n',
        f"All750 calibration events (1500 probe windows) and150 full-test cases passed independent count-to-fit/code, score and resource-clock audit. Separately constructed public THRML programs regenerate all1,550,400 active field hashes, replay every stored probe-segment count, and reconstruct every example in both neural layers. Maximum scaled preactivation/logit errors {replay['max_scaled_preactivation_error']:.3g}/{replay['max_scaled_logit_error']:.3g} are below2e-5 F32 summation tolerance. CPU/CUDA tests check same-path segment accumulation, reject premature ReLU and verify the complete nonoverlapping clock partition.\n",
        f"Production sampling totals {result['production_calibration_free_evaluations']:,} calibration and {result['production_inference_free_evaluations']:,} inference updates. Public replay repeats both as verification work, not extra independent accuracy realizations. Every source and exported model/input is hashed.\n",
        'The full long-horizon goal stays active for validation-selected quality/cost policies, training under nonideal devices, broader calibrated pretrained conversion and physical characterization. The fast-drift failure is retained.\n',
        '## Reproduction\n',
        '```bash\nOPENBLAS_NUM_THREADS=4 .venv/bin/python -m research.thrml_spw.gpu_environment -- .venv-thrml-gpu/bin/python -m research.thrml_spw.clocked_mnist\nOPENBLAS_NUM_THREADS=4 .venv/bin/python -m research.thrml_spw.gpu_environment -- .venv-thrml-gpu/bin/python -m research.thrml_spw.clocked_mnist_path_audit\nOPENBLAS_NUM_THREADS=4 .venv/bin/python -m research.thrml_spw.verify_clocked_mnist\n.venv-thrml/bin/python -m research.thrml_spw.clocked_mnist_report\n```\n'])
    rendered='\n'.join(text)+'\n'
    substitutions={'mean1':'mean 1','log-SD0.2':'log-SD 0.2','SD0.2':'SD 0.2','SD0,':'SD 0,','per1024':'per 1024','of313':'of 313','across32':'across 32','Layer0':'Layer 0','runs1024':'runs 1024','layer1':'layer 1','runs1024':'runs 1024','every64':'every 64','resetfalse':'reset false','A4096':'A 4096','batches0':'batches 0','contains2048':'contains 2048','by2048':'by 2048','field-1':'field -1','by2048':'by 2048','near+1':'near +1','predeclared[':'predeclared [','at681984':'at 681984',':641024':': 641024','plus40960':'plus 40960','spends4096':'spends 4096','and36864':'and 36864','spends40960':'spends 40960','is1.96':'is 1.96','last784':'last 784','require2368':'require 2368','All750':'All 750','and150':'and 150','all1,550,400':'all 1,550,400','below2e':'below 2e'}
    for a,b in substitutions.items():rendered=rendered.replace(a,b)
    Path('results/thrml_spw/CLOCKED_MNIST_REPORT.md').write_text(rendered)
    print(json.dumps({'summary':summary,'contrasts':contrasts,'resource_ledger':ledger},indent=2))


if __name__=='__main__':main()
