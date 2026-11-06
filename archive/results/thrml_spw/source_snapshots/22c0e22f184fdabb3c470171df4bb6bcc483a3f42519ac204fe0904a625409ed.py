"""Finalize completed native cases without rerunning or changing the frozen study.

The original driver imports the old128-window summarizer and will reject the new
64-window arrays only after its native phase finishes. This replacement finalizer
uses the same prospectively fixed crossed-seed/bank formula on the correct shape.
Run only after the original driver has authoritatively finished; it never changes
any native input/output, condition, budget, threshold or token reservation.
"""
import json
from pathlib import Path
import numpy as np
from .calibrated_transformer_statistics import summarize
from .transformer_block_gpu import sha,save_json,write_csv
OUT=Path('results/thrml_spw/calibrated_transformer')


def main():
    p=json.loads((OUT/'protocol.json').read_text());execution=json.loads((OUT/'execution_protocol.json').read_text());tokens=np.load(OUT/'tokens.npz')['tokens'];base=np.load(OUT/'baseline_tokens.npz')
    assert p['windows']==64 and p['conditions']==['ideal','uncalibrated','cal4096'] and sha(OUT/'protocol.json')==execution['prepared_design_sha256']
    for n,h in execution['source_sha256'].items():assert sha(n)==h
    rows=[];prefixes=[];profiles=[]
    for condition in p['conditions']:
        nl=[];pr=[]
        for seed in range(5):
            folder=OUT/'runs'/f'{condition}_seed{seed}';r=json.loads((folder/'run.json').read_text());assert r['status']=='complete' and r['design_sha256']==sha(OUT/'protocol.json') and len(r['stages'])==12
            for s in r['stages']:assert sha(s['output'])==json.loads(Path(s['metadata']).read_text())['output_sha256']
            a=np.load(folder/'prefix_tokens.npz');nl.append(a['prefix12_nll']);pr.append(a['prefix12_predictions'])
            rows.append({k:r[k] for k in ['split','condition','sampling_seed','free_node_updates','exact_prefix_input_checks']}|r['scores'])
            for stage in range(13):
                nll=a[f'prefix{stage}_nll'];pred=a[f'prefix{stage}_predictions'];ce=float(nll.astype(float).mean())
                prefixes.append({'condition':condition,'sampling_seed':seed,'converted_prefix':stage,'cross_entropy':ce,'perplexity':float(np.exp(ce)),'token_accuracy':float(np.mean(pred==tokens[:,1:]))})
        r,draws=summarize(np.stack(nl),np.stack(pr),base['FP32_nll'],base['FP32_predictions'],tokens);np.savez_compressed(OUT/f'{condition}_uncertainty_draws.npz',**draws);profiles.append({'condition':condition,**r})
    cal=list((OUT/'calibration').glob('*.json'));assert len(cal)==60
    correction={'reporting_only':True,'reason':'Original imported summary helper explicitly requires128 windows; frozen new design has64. Native phase and all raw case records are retained unchanged. New shape-specific helper uses identical declared crossed-seed/bank arithmetic and thresholds.',
        'original_helper_sha256':sha('research/thrml_spw/conservative_selection.py'),'original_driver_sha256':sha('research/thrml_spw/calibrated_transformer.py'),
        'finalizer_sha256':sha(__file__),'statistics_sha256':sha('research/thrml_spw/calibrated_transformer_statistics.py'),'raw_cases_changed':False,'new_native_sampling_performed':False}
    save_json(OUT/'postprocessing_correction.json',correction)
    write_csv(OUT/'native_metrics.csv',rows);write_csv(OUT/'prefix_metrics.csv',prefixes);write_csv(OUT/'summary.csv',profiles)
    manifest={'config':p,'execution':execution,'native_forward_runs':15,'native_operator_runs':180,'native_probe_windows':120,'profiles':profiles,
        'total_native_inference_free_evaluations':sum(r['free_node_updates'] for r in rows),'total_native_calibration_free_evaluations':sum(json.loads(c.read_text())['free_evaluations'] for c in cal),
        'postprocessing_correction':correction,'scope':p['scope']}
    save_json(OUT/'manifest.json',manifest);print(json.dumps({'profiles':profiles,'new_native_sampling_performed':False},indent=2))


if __name__=='__main__':main()
