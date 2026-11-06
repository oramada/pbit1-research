"""Train all ten declared RGB fits, then audit validation exports independently."""
import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
from results.thrml_spw.cifar_cnn_native.probability_finetune.train import ROOT, OUT, sha, write, guard


def main():
    p = json.loads((OUT/'protocol.json').read_text())
    guard(p)
    assert not (OUT/'pipeline.json').exists()
    assert shutil.disk_usage('.').free >= 50000000
    state = {'status': 'running', 'stages': []}
    prefix = 'results.thrml_spw.cifar_cnn_native.'
    for name, module, args in [('training', prefix+'probability_finetune.train', ['run']),
                               ('independent_validation', prefix+'verify_probability_finetune', [])]:
        guard(p)
        command = ['.venv/bin/python', '-m', module]+args
        row = {'stage': name, 'command': command, 'status': 'running',
               'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
        state['stages'].append(row)
        log = OUT/(name+'.log')
        with log.open('wb') as stream:
            child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                    env={**os.environ, 'OPENBLAS_NUM_THREADS': '4', 'OMP_NUM_THREADS': '4'})
            row['pid'] = child.pid
            write(OUT/'pipeline.json', state)
            code = child.wait()
        row.update(returncode=code, status='completed' if code == 0 else 'failed',
                   finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), log_sha256=sha(log))
        write(OUT/'pipeline.json', state)
        if code:
            state['status'] = 'failed'
            write(OUT/'pipeline.json', state)
            raise SystemExit(code)
    audit = json.loads((OUT/'evidence_audit.json').read_text())
    assert audit['passed'] and audit['optimizer_steps_recorded'] == 14080
    assert audit['selected_validation_logit_values_exact'] == 2000000
    manifest = json.loads((OUT/'manifest.json').read_text())
    assert manifest['complete'] and len(manifest['rows']) == 10 and not manifest['test_scored']
    report = ROOT.parent/'CIFAR_CNN_PROBABILITY_FINETUNE_PREPARATION_REPORT.md'
    assert not report.exists()
    lines = ['# RGB probability-aware fine-tuning preparation', '',
        'All five frozen RGB models/two declared methods/eight additional epochs completed with validation-only selection. Expected-weight and binomial-mean STE T256training share fixed centers/radii/biases/ACT8spans, model-specific minibatch orders and optimizer schedule. Only field latents are learned, with eight-bit forward commands and post-Adamprojection. All models/methods are retained, including epoch0if selected.', '',
        '| Model | Method | Selected epoch | Validation CE | Surrogate validation accuracy (%) |', '|---:|---|---:|---:|---:|']
    for row in audit['rows']:
        lines.append(f"|{row['model']}|{row['method']}|{row['selected_epoch']}|{row['validation_CE']:.8f}|{100*row['validation_accuracy']:.4f}|")
    lines += ['', 'Independent functional replay checked all2million selected validation logits exactly, all80minibatch orders/14080optimizer steps, fixed ranges/biases, command/field exports and all epoch-selection histories. This is not independent full retraining or native inference. Validation uses four fixed binomial-mean replicas atT256; training gradients are declared biased surrogates.', '',
        'The original frozenRGB2.6936point test gap/quality-target miss is retained. This exploratory follow-up uses additional training, so comparison to the original FP32baseline has a training confound; the expected-weight method is the matched parameterization/epoch/order training control. No test examples/labels enter optimization or selection, no fresh generalization or outcome-selected gate is asserted. Official native learned-command validation/full task/public/control/statistical confirmation remains required before a quality claim.', '',
        '[Frozen training protocol](cifar_cnn_native/probability_finetune_evidence/protocol.json) · [All fits](cifar_cnn_native/probability_finetune_evidence/manifest.json) · [Independent validation audit](cifar_cnn_native/probability_finetune_evidence/evidence_audit.json). The broad research goal remains active.']
    report.write_text('\n'.join(lines)+'\n')
    paths = [OUT/name for name in ['protocol.json', 'manifest.json', 'selfcheck.json', 'evidence_audit.json', 'training.log', 'independent_validation.log']]+[report]
    write(OUT/'completion.json', {'status': 'RGBprobability_training_validation_audited_goal_active',
          'recorded_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'artifact_sha256': {str(path): sha(path) for path in paths}})
    state['status'] = 'RGBprobability_training_validation_audited_goal_active'
    write(OUT/'pipeline.json', state)


if __name__ == '__main__':
    main()
