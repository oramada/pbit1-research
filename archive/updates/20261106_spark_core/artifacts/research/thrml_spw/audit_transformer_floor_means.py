"""Independent measured-fit provenance, error telescope and mean-network replay."""
import json,csv,hashlib
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from huggingface_hub.constants import HF_HUB_CACHE
from transformers import AutoModelForCausalLM
from .verify_transformer_control import sha
OUT=Path('results/thrml_spw/transformer_calibration_floor');OLD=Path('results/thrml_spw/calibrated_transformer')


def digest(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def main():
    torch.set_num_threads(8);torch.use_deterministic_algorithms(True)
    p=json.loads((OUT/'protocol.json').read_text());old=json.loads((OLD/'protocol.json').read_text());assert sha(OLD/'manifest.json')==p['source_manifest_sha256'] and sha(OLD/'evidence_audit.json')==p['source_evidence_audit_sha256'] and sha(OLD/'native_path_audit.json')==p['source_native_audit_sha256']
    for n,h in p['source_sha256'].items():assert sha(n)==h
    assert p['mean_conditions']==['ideal8','measured8','oracle8','oracle_continuous_bound4'] and p['native_runs']==5 and p['native_samples']==1024 and p['windows']==64
    assert sha(p['tokens_path'])==p['tokens_sha256'];tokens=np.load(p['tokens_path'])['tokens'];diagnostics=json.loads((OUT/'weight_diagnostics.json').read_text());assert len(diagnostics)==60
    with (OUT/'mean_metrics.csv').open() as f:rows=list(csv.DictReader(f))
    assert len(rows)==16;assert {(r['condition'],int(r['device_seed'])) for r in rows}=={('ideal8',0)}|{(c,s) for c in ['measured8','oracle8','oracle_continuous_bound4'] for s in range(5)}
    model=AutoModelForCausalLM.from_pretrained(Path(HF_HUB_CACHE)/f"models--distilbert--distilgpt2/snapshots/{p['revision']}",local_files_only=True,attn_implementation='eager').float().eval()
    modules=[model.get_submodule(n) for n in p['targets_in_forward_order']];original=[m.weight.detach().clone() for m in modules];master=np.load(old['master_weights_path']);assert sha(old['master_weights_path'])==old['master_weights_sha256']
    max_closure=0.;checked=0;field_hashes=0
    for seed in range(5):
        weights={method:[] for method in p['mean_conditions']};matrixhash={method:[] for method in p['mean_conditions']}
        for stage,c in enumerate(p['control_artifacts']):
            assert sha(c['path'])==c['sha256'];control=np.load(c['path']);shape=control['codes'].shape;size=int(np.prod(shape));b=(-4+control['codes'].reshape(-1).astype(float)*8/255).astype(np.float32)
            noise=np.random.default_rng(np.random.SeedSequence([2026114600,stage,seed])).standard_normal((2,size));gain=np.exp(.2*noise[1]-.02);offset=.2*noise[0]
            cp=OLD/'calibration'/f'stage{stage}_seed{seed}.npz';a=np.load(cp);cm=json.loads(cp.with_suffix('.json').read_text());assert sha(cp)==cm['archive_sha256']
            # Recheck that measured codes come exclusively from old actual counts.
            low,high=a['count0'].astype(float),a['count1'].astype(float);pf=(-4+np.array([96,159])*8/255).astype(np.float32).astype(float)
            ul=.5*(np.log(low+.5)-np.log(2048-low+.5));uh=.5*(np.log(high+.5)-np.log(2048-high+.5));raw=(uh-ul)/(pf[1]-pf[0]);ah=np.clip(raw,.25,4);dh=(ul+uh)/2-raw*(pf[1]+pf[0])/2
            measured=np.rint((np.clip((b.astype(float)-dh)/ah,-4,4)+4)*255/8).astype(np.uint8);np.testing.assert_array_equal(measured,a['corrected_codes'])
            inv=(b.astype(float)-offset)/gain;oracle=np.rint((np.clip(inv,-4,4)+4)*255/8).astype(np.uint8)
            measured_cmd=(-4+measured.astype(float)*8/255).astype(np.float32);oracle_cmd=(-4+oracle.astype(float)*8/255).astype(np.float32)
            fs={'ideal8':b,'measured8':(gain*measured_cmd.astype(float)+offset).astype(np.float32),'oracle8':(gain*oracle_cmd.astype(float)+offset).astype(np.float32),'oracle_continuous_bound4':(gain*np.clip(inv,-4,4)+offset).astype(np.float32)}
            row=next(r for r in diagnostics if r['stage']==stage and r['seed']==seed);assert row['calibration_archive_sha256']==sha(cp) and row['true_inverse_bound_clipped']==np.count_nonzero(np.abs(inv)>4) and row['oracle_code_sha256']==digest(oracle)
            q={k:1/(1+np.exp(-2*f.astype(float))) for k,f in fs.items()};sp=np.repeat(control['span'],128,0).reshape(-1);lo=np.repeat(control['lower'],128,0)
            effects=[sp*(q['measured8']-q['oracle8']),sp*(q['oracle8']-q['oracle_continuous_bound4']),sp*(q['oracle_continuous_bound4']-q['ideal8'])];total=sp*(q['measured8']-q['ideal8'])
            np.testing.assert_allclose(sum(effects),total,atol=1e-14,rtol=1e-12)
            vals={'weights':size,'total_SSE':float(np.sum(total**2)),'measured_minus_known_SSE':float(np.sum(effects[0]**2)),'oracle_grid_SSE':float(np.sum(effects[1]**2)),'oracle_range_SSE':float(np.sum(effects[2]**2)),
                'twice_estimate_grid':float(2*np.sum(effects[0]*effects[1])),'twice_estimate_range':float(2*np.sum(effects[0]*effects[2])),'twice_grid_range':float(2*np.sum(effects[1]*effects[2]))}
            vals['SSE_closure_error']=sum(vals[k] for k in ['measured_minus_known_SSE','oracle_grid_SSE','oracle_range_SSE','twice_estimate_grid','twice_estimate_range','twice_grid_range'])-vals['total_SSE'];max_closure=max(max_closure,abs(vals['SSE_closure_error']))
            for k,v in vals.items():np.testing.assert_allclose(row['moments'][k],v,atol=1e-12,rtol=1e-11)
            for method in p['mean_conditions']:
                assert row['field_sha256'][method]==digest(fs[method]);field_hashes+=1
                full=lo+sp.reshape(shape)*q[method].reshape(shape);value=full.astype(np.float32);assert row['mean_matrix_sha256'][method]==digest(value)
                np.testing.assert_allclose(row['target_MSE'][method],np.mean((full-master[f'weight_{stage}'].astype(float))**2),atol=1e-15,rtol=1e-11)
                weights[method].append(value);matrixhash[method].append(digest(value))
        for method in p['mean_conditions']:
            if method=='ideal8' and seed:continue
            path=OUT/'mean_runs'/f'{method}_seed{seed}.npz';meta=json.loads(path.with_suffix('.json').read_text());saved=np.load(path);assert sha(path)==meta['archive_sha256'] and meta['mean_matrix_sha256']==matrixhash[method]
            try:
                with torch.no_grad():
                    for module,w in zip(modules,weights[method]):module.weight.copy_(torch.from_numpy(w))
                nll=[];pred=[]
                with torch.inference_mode():
                    for row in tokens:
                        ids=torch.from_numpy(row[None]);logits=model(input_ids=ids,use_cache=False).logits[:,:-1];target=ids[:,1:]
                        nll.append(F.cross_entropy(logits.reshape(-1,logits.shape[-1]),target.reshape(-1),reduction='none').numpy());pred.append(logits.argmax(-1).numpy()[0])
                np.testing.assert_array_equal(saved['nll'],np.stack(nll));np.testing.assert_array_equal(saved['predictions'],np.stack(pred))
                ce=float(saved['nll'].astype(float).mean());pp=float(np.exp(ce));accuracy=float(np.mean(saved['predictions']==tokens[:,1:]))
                csvrow=next(r for r in rows if r['condition']==method and int(r['device_seed'])==seed)
                for key,value in [('cross_entropy',ce),('perplexity',pp),('token_accuracy',accuracy)]:assert meta[key]==value;np.testing.assert_allclose(float(csvrow[key]),value,atol=1e-14,rtol=1e-13)
                assert meta['native_free_evaluations']==0;checked+=1
            finally:
                with torch.no_grad():
                    for module,w in zip(modules,original):module.weight.copy_(w)
            print(f'Independent analytic mean-model replay {method} seed{seed}',flush=True)
        del weights
    assert checked==16 and field_hashes==240
    audit={'passed':True,'mean_forward_runs':16,'weight_field_counterfactuals_checked':240,'paired_stage_device_moment_decompositions':60,'max_absolute_SSE_closure_error':max_closure,'measured_codes_independently_refit_from_hashed_original_native_counts':True,'all_mean_model_token_losses_and_predictions_replayed':True,'source_sha256':sha(__file__),
        'scope':'Analytic programmed-mean counterfactuals and exact signed-cross-term weight-error identity. No new stochastic samples, no expectation-of-nonlinear-network or deployable-oracle claim. Native oracle performance checked separately.'}
    (OUT/'mean_evidence_audit.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps(audit,indent=2))


if __name__=='__main__':main()
