"""Bounded calibration/drift evidence synthesis; no new sampling or training.

Recompute quarter-model mean losses/accuracy from all saved final token arrays,
then summarize all clocked-drift seed rows with explicit calibration cost units.
Existing independent raw-stage/path audits remain the native evidence boundary.
"""
from pathlib import Path
import csv,hashlib,json,math
import numpy as np

BASE=Path('results/thrml_spw')
OUT=BASE/'broad_verification_20261105/nonideality_evidence_v1'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def table(p):
    with Path(p).open()as f:return list(csv.DictReader(f))

def close(x,y):np.testing.assert_allclose(x,y,rtol=1e-11,atol=1e-12)


def main():
    inputs={}
    def bind(path):inputs[str(path)]=sha(path);return path
    def read(path):return json.loads(bind(path).read_text())
    cal=BASE/'calibrated_transformer';drift=BASE/'drift_retention_operator'
    ca=read(cal/'evidence_audit.json');assert ca['passed'] and ca['all_native_nodes_independently_replayed']
    assert ca['manifest_sha256']==sha(bind(cal/'manifest.json'))
    for key,name in [('model_score_audit_sha256','model_score_audit.json'),('native_path_audit_sha256','native_path_audit.json'),('hook_audit_sha256','independent_hook_audit.json')]:assert ca[key]==sha(bind(cal/name))
    assert read(cal/'source_lineage_audit.json')['passed']
    with np.load(bind(cal/'tokens.npz'),allow_pickle=False)as a:targets=a['tokens'][:,1:].copy()
    with np.load(bind(cal/'baseline_tokens.npz'),allow_pickle=False)as a:
        baseline_ce=float(a['FP32_nll'].astype(np.float64).mean());baseline_acc=float(np.mean(a['FP32_predictions']==targets))
    assert targets.shape==(64,64)
    baseline_rows=table(bind(cal/'baseline_metrics.csv'));bf=next(x for x in baseline_rows if x['condition']=='FP32')
    close(baseline_ce,float(bf['cross_entropy']));close(math.exp(baseline_ce),float(bf['perplexity']));close(baseline_acc,float(bf['token_accuracy']))
    rows=table(bind(cal/'summary.csv'));assert [r['condition']for r in rows]==['ideal','uncalibrated','cal4096']
    cal_conditions=[]
    for row in rows:
        name=row['condition'];losses=[];accuracy=[]
        for seed in range(5):
            with np.load(bind(cal/'runs'/f'{name}_seed{seed}'/'prefix_tokens.npz'),allow_pickle=False)as a:
                assert a['prefix12_nll'].shape==targets.shape;losses.append(float(a['prefix12_nll'].astype(np.float64).mean()))
                accuracy.append(float(np.mean(a['prefix12_predictions']==targets)))
        ce=math.fsum(losses)/5;acc=math.fsum(accuracy)/5;ratio=math.exp(ce-baseline_ce)
        close(ce,float(row['cross_entropy_mean']));close(acc,float(row['token_accuracy_mean']));close(ratio,float(row['relative_perplexity']))
        point=(ratio<=1.05 and acc-baseline_acc>=-.01)
        assert point==(row['point_tolerance_passed']=='True')
        upper=float(row['bootstrap_relative_upper']);lower=float(row['bootstrap_accuracy_difference_lower'])
        assert (upper<=1.05 and lower>=-.01)==(row['passes']=='True')
        cal_conditions.append({'condition':name,'mean_NLL':ce,'perplexity':math.exp(ce),'perplexity_ratio_to_FP32':ratio,
            'per_seed_NLL':losses,'per_seed_accuracy':accuracy,'accuracy_gap_pp':100*(acc-baseline_acc),
            'point_gate_met':point,'original_crossed_bootstrap_ratio_upper':upper,
            'original_crossed_bootstrap_accuracy_lower_pp':100*lower,'original_crossed_gate_met':row['passes']=='True'})
    ledger=read(cal/'resource_ledger.json');assert ledger['converted_weights']==21233664 and ledger['model_unique_parameters']==81912576
    assert ledger['measured_calibration_free_evaluations_per_profile']==86973087744
    assert ledger['inference_free_evaluations_per_64context_profile']==173946175488
    close(ledger['observed_evaluation_multiplier_calibrated_vs_uncalibrated'],1.5)
    da=read(drift/'evidence_audit.json');assert da['passed'] and da['native_query_windows']==150 and da['native_calibration_events']==90
    assert da['clock_and_calibration_idle_costs_verified'] and read(drift/'source_lineage_audit.json')['passed']
    manifest=read(drift/'manifest.json');protocol=read(drift/'protocol.json');production=read(drift/'REPORT.json')
    assert manifest['config']==protocol and manifest['result']==production
    assert production['free_nodes']==2322432 and production['native_probe_windows']==180
    rows=table(bind(drift/'query_metrics.csv'));assert len(rows)==150;groups=[];contrasts=[]
    for sigma in [0.,.01,.04]:
        for age in [0,8,32]:
            policies=['retain']if age==0 else['retain','refresh'];by_policy={}
            for policy in policies:
                rr=sorted([r for r in rows if float(r['sigma'])==sigma and int(r['age'])==age and r['policy']==policy],key=lambda x:int(x['seed']))
                assert [int(r['seed'])for r in rr]==list(range(10))
                values=np.array([float(r['observed_total_mse'])for r in rr]);by_policy[policy]=values
                event_count=2 if policy=='refresh'else 1
                assert {int(r['counterfactual_calibration_events'])for r in rr}=={event_count}
                assert {int(r['counterfactual_calibration_free_evaluations'])for r in rr}=={2322432*4096*event_count}
                assert {int(r['inference_free_node_evaluations'])for r in rr}=={2322432*512}
                clocks={(int(r['counterfactual_active_sweeps']),int(r['counterfactual_idle_sweeps']),int(r['query_end_clock_sweeps']))for r in rr}
                assert len(clocks)==1;active,idle,end=clocks.pop();assert active+idle==end and active==4096*event_count+512
                groups.append({'drift_SD_per_512sweep_interval':sigma,'age_intervals':age,'policy':policy,
                    'seeds':list(range(10)),'mean_MSE':float(values.mean()),'approx_normal95_halfwidth':float(1.96*values.std(ddof=1)/math.sqrt(10)),
                    'per_seed_MSE':values.tolist(),'calibration_events':event_count,
                    'calibration_free_node_evaluations':2322432*4096*event_count,'query_free_node_evaluations':2322432*512,
                    'active_sweeps':active,'idle_sweeps':idle,'query_end_clock_sweeps':end})
            if age:
                diff=by_policy['refresh']-by_policy['retain']
                contrasts.append({'drift_SD_per_512sweep_interval':sigma,'age_intervals':age,'refresh_minus_retain_mean_MSE':float(diff.mean()),
                    'paired_approx_normal95_halfwidth':float(1.96*diff.std(ddof=1)/math.sqrt(10)),
                    'refresh_to_retain_mean_MSE_ratio':float(by_policy['refresh'].mean()/by_policy['retain'].mean())})
    assert len(groups)==15 and len(contrasts)==6
    assert all(sha(p)==h for p,h in inputs.items())
    result={'status':'existing_bounded_calibration_and_drift_summaries_recomputed_and_hash_bound','source_sha256':sha(__file__),'inputs_sha256':inputs,
        'calibration':{'baseline_FP32_perplexity':math.exp(baseline_ce),'baseline_accuracy':baseline_acc,'conditions':cal_conditions,
            'converted_parameters':21233664,'full_model_parameters':81912576,'native_samples':1024,'calibration_samples_per_free_weight':4096,
            'numeric_control_range_bias_payload_bits_per_weight':ledger['numeric_payload_bits_per_converted_weight'],
            'calibration_free_updates_per_profile':86973087744,'inference_free_updates_per_profile':173946175488,
            'scope':'Selected first3DistilGPT2blocks,25.922percentparameterconversion;8bitfinitecontrols,5syntheticstaticdevice/samplingseeds,64shortcontexts/4096targets. R2operator accumulation; otherweights,biases,normsFP32. Fixed5percentPPL/1ppaccuracygates, unlike newer full-model2percentBF16study. Calibration improvesqualitybutstillfailsbothoriginalgates. No new trials or full-model/hardwareclaim.'},
        'drift':{'operator_weights':2359296,'free_nodes':2322432,'groups':groups,'paired_contrasts':contrasts,
            'scope':'Old8developmentcontexts,one2.36Mnormalizedweightoperator,10jointdevice/path/samplingseeds. SyntheticBrownianoffsetdriftthroughprobesandinference;agein512sweepintervals,notseconds. Clampednodesexcludedfromfreeupdates. Counterfactualrefreshandretainqueryatsameabsoluteclock. MSEandnormalintervalsnotfull-modelqualityorhardwarecalibrationlifetime.'},
        'physical_hardware_energy_J':None,'physical_hardware_latency_s':None,
        'scope':'No new stochastic or neuralforward executions. Meanloss/predictions independentlyrecomputedfromsavedtokenarrays; driftseedtables/costidentitiesrecomputed. Nativebit/pathprovenance relies on separatelyboundexistingindependentaudits. Does not establishadditionalfull-modelrobustnessunderdeviceerrors.'}
    OUT.mkdir(exist_ok=False);(OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'calibration_gate_results':[(r['condition'],r['original_crossed_gate_met'])for r in cal_conditions],'drift_conditions':len(groups),'paired_contrasts':len(contrasts)}))

if __name__=='__main__':main()
