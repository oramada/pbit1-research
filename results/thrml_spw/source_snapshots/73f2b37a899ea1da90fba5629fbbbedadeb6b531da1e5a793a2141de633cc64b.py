"""Five-seed probability-training evidence, validation selections and task controls."""
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path('results/thrml_spw/probability_finetune');KINDS=['initial','expected','native_ideal','native_device'];CONDS=['ideal','uncalibrated','cal4096']


def read(path):
    with path.open() as f:return list(csv.DictReader(f))


def margin(x):return float(1.96*np.std(x,ddof=1)/np.sqrt(len(x)))


def main():
    audit=json.loads((OUT/'evidence_audit.json').read_text());assert audit['passed'];training=json.loads((OUT/'TRAINING_REPORT.json').read_text());test=json.loads((OUT/'TEST_REPORT.json').read_text())
    ta=json.loads((OUT/'training_path_audit.json').read_text());te=json.loads((OUT/'test_path_audit.json').read_text());selection=json.loads((OUT/'selected.json').read_text())
    rows=read(OUT/'test_metrics.csv');base=read(OUT/'test_baseline_metrics.csv');val=read(OUT/'validation_metrics.csv');summary=[];vectors={};classical=[];contrasts=[]
    for kind in KINDS:
        for condition in CONDS:
            for t in [64,256]:
                a=np.array([np.mean([float(r['accuracy']) for r in rows if int(r['model'])==m and r['kind']==kind and r['condition']==condition and int(r['T'])==t]) for m in range(5)])
                ce=np.array([np.mean([float(r['cross_entropy']) for r in rows if int(r['model'])==m and r['kind']==kind and r['condition']==condition and int(r['T'])==t]) for m in range(5)])
                vectors[kind,condition,t]=a
                summary.append({'kind':kind,'condition':condition,'T':t,'accuracy':float(a.mean()),'model_normal95_margin':margin(a),'cross_entropy':float(ce.mean()),'models':5,'reserved_device_sampling_seeds_per_model':5})
        for condition in ['original_FP32','expected','UINT4','UINT8']:
            a=np.array([float(next(r['accuracy'] for r in base if int(r['model'])==m and r['kind']==kind and r['condition']==condition)) for m in range(5)])
            classical.append({'kind':kind,'condition':condition,'accuracy':float(a.mean()),'model_normal95_margin':margin(a)})
    for kind,reference in [('native_device','expected'),('native_ideal','expected'),('native_device','native_ideal'),('native_device','initial')]:
        for condition in CONDS:
            for t in [64,256]:
                d=vectors[kind,condition,t]-vectors[reference,condition,t]
                contrasts.append({'kind':kind,'reference':reference,'condition':condition,'T':t,'paired_accuracy_difference':float(d.mean()),'paired_model_normal95_margin':margin(d)})
    for name,data in [('summary.csv',summary),('classical_summary.csv',classical),('paired_contrasts.csv',contrasts)]:
        with (OUT/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    primary=next(r for r in contrasts if r['kind']=='native_device' and r['reference']=='expected' and r['condition']=='uncalibrated' and r['T']==64)
    resources={'native_training_free_evaluations':training['native_training_free_evaluations'],'native_validation_free_evaluations':training['native_validation_free_evaluations'],
        'native_test_free_evaluations':test['native_test_free_evaluations'],'native_shared_test_calibration_free_evaluations':test['native_test_calibration_free_evaluations'],
        'independent_replay_native_free_evaluations':sum([training['native_training_free_evaluations'],training['native_validation_free_evaluations'],test['native_test_free_evaluations'],test['native_test_calibration_free_evaluations']]),
        'native_training_steps':10750,'expected_training_steps':5375,'weights_per_model':2368,'logical_free_weight_pbits_per_model':2368,'clamped_nodes':0,'deterministic_neural_biases':42,
        'native_train_samples_per_step':64,'train_batch_size':256,'target_commands_uint8_bytes_per_checkpoint':2368,
        'training_float32_theta_bytes':2368*4,'training_two_float32_Adam_moments_bytes':2*2368*4,
        'candidate_calibration_free_evaluations_per_device_per_checkpoint_if_used':4096*2368,
        'scope':'Production calibration counts shared across four checkpoint conditions because true physical synthetic devices and probes are fixed independently of learned targets. Candidate inference cost includes its own calibration where used. Replay adds verification work, not independent training seeds. Floating software training/storage and external neural arithmetic remain; no physical energy/speed claim.'}
    (OUT/'resource_ledger.json').write_text(json.dumps(resources,indent=2)+'\n')
    fig,ax=plt.subplots(2,2,figsize=(12,8),layout='constrained');xx=np.arange(4);labels=['Initial','Expected\ntraining','Native ideal\ntraining','Native device\ntraining']
    for t in [64,256]:
        g=[next(r for r in summary if r['kind']==k and r['condition']=='uncalibrated' and r['T']==t) for k in KINDS]
        ax[0,0].errorbar(xx,[r['accuracy']*100 for r in g],yerr=[r['model_normal95_margin']*100 for r in g],fmt='o-',label=f'T{t}')
    ax[0,0].set(title='Reserved uncalibrated devices',xticks=xx,xticklabels=labels,ylabel='Accuracy (%)');ax[0,0].legend(fontsize=8)
    for condition in ['ideal','cal4096']:
        g=[next(r for r in summary if r['kind']==k and r['condition']==condition and r['T']==256) for k in KINDS]
        ax[0,1].errorbar(xx,[r['accuracy']*100 for r in g],yerr=[r['model_normal95_margin']*100 for r in g],fmt='o-',label=condition+' T256')
    ax[0,1].set(title='Ideal and measured-calibrated controls',xticks=xx,xticklabels=labels,ylabel='Accuracy (%)');ax[0,1].legend(fontsize=8)
    for kind in KINDS[1:]:
        values=[]
        for epoch in range(6):values.append(np.mean([float(r['cross_entropy']) for r in val if r['method']==kind and int(r['epoch'])==epoch]))
        ax[1,0].plot(np.arange(6),values,'o-',label=kind.replace('_',' '))
    ax[1,0].set(title='Validation only: fixed profiles and MC keys',xlabel='Epoch (0 = finite initial probabilities)',ylabel='Mean native CE at T64');ax[1,0].legend(fontsize=8)
    for m in range(5):
        eps=[selection[f'model{m}_{k}']['selected_epoch'] for k in KINDS[1:]]
        ax[1,1].plot(np.arange(3)+(m-2)*.06,eps,'o',label=f'Model {m}')
    ax[1,1].set(title='Selected before test evaluation',xticks=np.arange(3),xticklabels=labels[1:],yticks=np.arange(6),ylabel='Selected epoch');ax[1,1].legend(fontsize=7,ncol=2)
    for a in ax.ravel():a.grid(axis='y',alpha=.2)
    fig.savefig('results/thrml_spw/probability_finetune_evidence.png',dpi=180);plt.close(fig)
    native=vectors['native_device','uncalibrated',64].mean();expected=vectors['expected','uncalibrated',64].mean();fp=next(r['accuracy'] for r in classical if r['kind']=='initial' and r['condition']=='original_FP32')
    text=['# Five-seed native probability fine-tuning\n',
        f"The predeclared primary comparison scores {100*native:.3f}% after native device-augmented probability fine-tuning versus {100*expected:.3f}% after expected-weight fine-tuning on reserved uncalibrated synthetic devices at T64. The paired difference is {100*primary['paired_accuracy_difference']:+.3f} percentage points with approximate five-model normal margin {100*primary['paired_model_normal95_margin']:.3f}. Original FP32 accuracy is {100*fp:.3f}%; noisy-device inference is a different condition and should not be equated with deterministic accuracy.\n",
        '## Fixed design\n',
        'Five independently pretrained positive-weight 64->32->10 MNIST networks each initialize three paired probability fine-tuning methods: expected-weight, native ideal-device and native device-augmented. These are fifteen fine-tuning runs from five distinct starting models, not fifteen independent initializations or training from scratch. The 42 neural biases are frozen. Only the 2,368 probability logits theta are optimized, with p=sigmoid(theta), Adam lr0.001 and projection theta into[-8,8]. Initial endpoint weights are replaced by finite probabilities; no exact clamp nodes occur.\n',
        'All methods execute five epochs over the same seed/epoch-specific shuffled55,000 training examples, batch256 with40 repeated padding examples per epoch. Native methods use actual official THRML samples T64 per layer, with linear accumulation before ReLU and independent layer keys. Native device augmentation redraws synthetic static gain/offset once per minibatch, holding it throughout that minibatch\'s draws. Gain is lognormal mean1/log-SD0.2 and offset SD0.2. This models distribution robustness, not a continuously drifting device or measured hardware. Native ideal uses unit gain/zero offset; expected-weight training does not sample.\n',
        'The native forward emits nearest eight-bit bias commands in[-4,4] using F64 control/device arithmetic followed by F32 fields. Backward uses the explicitly biased expected-linear surrogate through unquantized sigmoid probabilities, including the expected second-layer matrix. It does not differentiate the discrete sampler, control grid or device parameters. This is a straight-through estimator, not an unbiased noisy-task gradient. Optimizer and programming precision are separate from one free p-bit per matrix weight.\n',
        '## Selection and reserved evaluation\n',
        'Each method/model selects minimum mean native uncalibrated validation CE across five fixed profiles on all5,000 training-held-out validation images, among epochs0--5 including the unchanged finite initial checkpoint. T64, validation noise/profiles, learning rate and epoch schedule are fixed before results. Earliest epoch breaks a tie. Common validation MC keys reduce comparison noise but permit selection overfitting to that finite bank; test uses separate profile-domain seeds and MC key roots. Validation is not a new dataset.\n',
        'The selection hash is frozen before the evaluation program loads test arrays or scores. All600 full-test cases cover the initial finite checkpoint and three selected methods, five models, five reserved static device/sampling realizations, ideal/uncalibrated/measured-cal4096 conditions and T64/T256. The entire10,000-image existing MNIST test set is evaluated, batch32, with16 zero-padding rows discarded. These images were used in earlier studies, so this is independent stochastic/device confirmation rather than fresh-data generalization. No test-driven retuning or favorable-run selection occurs.\n',
        'Measured calibration uses actual2,048 native draws per probe at finite codes96/159, a Jeffreys-smoothed two-probe gain/offset fit and declared gain guard[0.25,4]. It never receives true gain/offset or labels. Probe evidence is shared across checkpoint targets on the same devices, since probe fields do not depend on those targets. Programs use one free SpinNode per weight and external floating neural arithmetic. No drift, temporal/cross-device persistence or readout error is added in this training study.\n',
        '## All native task results\n',
        '| Checkpoint | Device condition | T/layer | Accuracy (%) | Model normal margin (pp) | CE |\n|---|---|---:|---:|---:|---:|']
    for r in summary:text.append(f"| {r['kind']} | {r['condition']} | {r['T']} | {100*r['accuracy']:.3f} | {100*r['model_normal95_margin']:.3f} | {r['cross_entropy']:.4f} |")
    text.extend(['\nAll paired contrasts are retained in paired_contrasts.csv. Five reserved device/sampling realizations are averaged within each starting-model seed first; displayed uncertainty is1.96*SD/sqrt(5) over the five model means. Contrasts use model-level paired differences. Small-sample normal margins are descriptive, not formal coverage guarantees or evidence for twenty-five independent trained models.\n',
        '## Deterministic and conventional low-bit controls\n',
        '| Checkpoint | Numerical control | Accuracy (%) | Model normal margin (pp) |\n|---|---|---:|---:|'])
    for r in classical:text.append(f"| {r['kind']} | {r['condition']} | {100*r['accuracy']:.3f} | {100*r['model_normal95_margin']:.3f} |")
    text.extend(['\nThe expected diagnostic uses unquantized learned probability weights in conventional arithmetic. UINT4/UINT8 use deterministic per-layer unsigned quantization of each checkpoint\'s learned probability weights. They are numerical controls, not matched energy/latency hardware. Full-data expected inference is not expectation of the noisy ReLU network. Improvements on noisy devices do not establish superiority over these controls, and any ideal/deterministic quality tradeoff remains visible.\n',
        '## Selected epochs\n',
        '| Model | Expected | Native ideal | Native device |\n|---:|---:|---:|---:|'])
    for m in range(5):text.append('| '+str(m)+' | '+' | '.join(str(selection[f'model{m}_{k}']['selected_epoch']) for k in KINDS[1:])+' |')
    text.extend(['\n## Independent training and inference evidence\n',
        f"Every one of10,750 native training updates and all450 complete validation/profile evaluations are replayed through separately constructed public THRML state-frequency programs. Every training forward (including5,375 expected-control updates), explicit biased-STE derivative and Adam update is checked against saved minibatch/theta/gradient records. Maximum scaled forward error is {ta['max_scaled_forward_error']:.3g}; maximum one-step Adam error is {ta['max_scaled_one_step_Adam_error']:.3g}. The gradient check uses fixed tolerance2e-6+2e-4*abs(reference), not an unbiasedness claim. Validation selection is independently recomputed, including allowed epoch0 choices.\n",
        f"All600 reserved full-test cases and50 native probe windows are independently replayed and scored. Maximum scaled test-layer error is {te['max_scaled_forward_error']:.3g}, below2e-5 tolerance for F32 summation/reassociation. All codes, measured fits, conventional controls, full scores and update counters are independently checked. CPU/CUDA golden tests verify native paths, surrogate derivatives, profile-domain separation and optimizer bounds. Archived source versions, data, checkpoints, gradients, optimizer states, validation outputs and test outputs support exact provenance.\n",
        '## Resource ledger and limits\n',
        f"Native training executes {resources['native_training_free_evaluations']:,} free-node draws, validation {resources['native_validation_free_evaluations']:,}, reserved test inference {resources['native_test_free_evaluations']:,}, and shared test calibration {resources['native_shared_test_calibration_free_evaluations']:,}. Independent replay repeats this native work for verification; it does not add independent training realizations. Each candidate calibrated deployment must count its calibration even though the experiment shares probe evidence across target checkpoints.\n",
        'A trained model has2,368 free weight nodes, zero exact endpoint modes and42 deterministic neural biases. Eight-bit commands occupy2,368 bytes, while float32 theta and two Adam states alone require28,416 bytes during training, before gradients, activations, graphs and saved histories. The simulator does not demonstrate one total storage bit per learned weight or a physical training/inference energy, area or latency advantage. Synthetic distribution assumptions and positive-weight model capacity limit generalization.\n',
        'The broad goal stays active for other gradient estimators, training with measured/clocked device behavior, broader architectures and calibrated pretrained conversion, validation-selected execution budgets and physical-system characterization. Earlier failures remain preserved.\n',
        '## Reproduction\n',
        '```bash\nOPENBLAS_NUM_THREADS=4 .venv/bin/python -m research.thrml_spw.gpu_environment -- .venv-thrml-gpu/bin/python -m research.thrml_spw.probability_finetune\nOPENBLAS_NUM_THREADS=4 .venv/bin/python -m research.thrml_spw.gpu_environment -- .venv-thrml-gpu/bin/python -m research.thrml_spw.audit_probability_finetune --phase training\nOPENBLAS_NUM_THREADS=4 .venv/bin/python -m research.thrml_spw.gpu_environment -- .venv-thrml-gpu/bin/python -m research.thrml_spw.probability_finetune_evaluate\nOPENBLAS_NUM_THREADS=4 .venv/bin/python -m research.thrml_spw.gpu_environment -- .venv-thrml-gpu/bin/python -m research.thrml_spw.audit_probability_finetune --phase test\n.venv-thrml/bin/python -m research.thrml_spw.probability_finetune_report\n```\n'])
    rendered='\n'.join(text)+'\n'
    for a,b in {'into[':'into [','shuffled55':'shuffled 55','batch256':'batch 256','with40':'with 40','mean1':'mean 1','log-SD0.2':'log-SD 0.2','SD0.2':'SD 0.2','in[-4':'in [-4','all5,000':'all 5,000','epochs0':'epochs 0','All600':'All 600','entire10,000':'entire 10,000','batch32':'batch 32','with16':'with 16','actual2,048':'actual 2,048','codes96':'codes 96','guard[':'guard [','is1.96':'is 1.96','of10,750':'of 10,750','all450':'all 450','including5,375':'including 5,375','tolerance2e':'tolerance 2e','epoch0':'epoch 0','and50':'and 50','below2e':'below 2e','has2,368':'has 2,368','and42':'and 42','occupy2,368':'occupy 2,368','require28,416':'require 28,416'}.items():rendered=rendered.replace(a,b)
    Path('results/thrml_spw/PROBABILITY_FINETUNE_REPORT.md').write_text(rendered)
    print(json.dumps({'primary':primary,'summary':summary,'classical':classical,'resources':resources},indent=2))


if __name__=='__main__':main()
