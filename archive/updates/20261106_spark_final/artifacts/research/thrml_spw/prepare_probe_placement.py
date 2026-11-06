"""Freeze weight-only probe placements and new confirmation positions, no scores."""
import json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from huggingface_hub.constants import HF_HUB_CACHE
from transformers import AutoTokenizer
from .probe_placement_math import design
from .verify_transformer_control import sha
OLD=Path('results/thrml_spw/calibrated_transformer');OUT=Path('results/thrml_spw/probe_placement')


def main():
    OUT.mkdir(exist_ok=True)
    if (OUT/'protocol.json').exists():raise RuntimeError('Prepared immutable study already exists')
    source=json.loads((OLD/'protocol.json').read_text());assert json.loads((OLD/'evidence_audit.json').read_text())['passed']
    placements=[]
    for c in source['control_artifacts']:
        assert sha(c['path'])==c['sha256'];a=np.load(c['path']);placements.append({'stage':c['stage'],**design(a['codes'],a['span'])})
    (OUT/'placement_designs.json').write_text(json.dumps(placements,indent=2)+'\n')
    excluded=source['excluded_archives']+[{'path':str(OLD/'tokens.npz'),'key':'token_offsets','sha256':sha(OLD/'tokens.npz')}]
    used=np.concatenate([np.load(e['path'])[e['key']] for e in excluded]+[np.array(source['legacy_excluded_offsets'])])
    parquet=Path(hf_hub_download('Salesforce/wikitext','wikitext-2-raw-v1/test-00000-of-00001.parquet',repo_type='dataset',revision=source['dataset_revision'],local_files_only=True));assert sha(parquet)==source['source_parquet_sha256']
    tokenizer=AutoTokenizer.from_pretrained(Path(HF_HUB_CACHE)/f"models--distilbert--distilgpt2/snapshots/{source['revision']}",local_files_only=True)
    texts=pq.read_table(parquet,columns=['text']).column('text').to_pylist();ids=np.asarray(tokenizer('\n'.join(t for t in texts if t and t.strip()),add_special_tokens=False)['input_ids'],np.int64)
    eligible=np.arange(len(ids)//65)*65;eligible=eligible[~np.any((eligible[:,None]<used[None]+65)&(used[None]<eligible[:,None]+65),axis=1)]
    starts=np.sort(np.random.default_rng(2026116700).choice(eligible,64,replace=False));tokens=np.stack([ids[i:i+65] for i in starts]);np.savez_compressed(OUT/'tokens.npz',tokens=tokens,token_offsets=starts)
    p={'schema':'spw-weight-only-probe-placement-confirmation-v1','revision':source['revision'],'dataset_revision':source['dataset_revision'],'source_parquet_path':str(parquet),'source_parquet_sha256':sha(parquet),
        'tokens_sha256':sha(OUT/'tokens.npz'),'token_selection_seed':2026116700,'excluded_archives':excluded,'legacy_excluded_offsets':source['legacy_excluded_offsets'],'windows':64,'targets':4096,
        'targets_in_forward_order':source['targets_in_forward_order'],'control_artifacts':source['control_artifacts'],'converted_weights':21233664,'model_unique_parameters':81912576,
        'placement_designs_sha256':sha(OUT/'placement_designs.json'),'methods':['fixed','weight_fisher'],'device_seeds':list(range(5)),'samples_each_probe':2048,'calibration_samples':4096,'native_samples':1024,'contexts_per_bank':8,
        'selection':'Perstage choose minimum nominaldelta/Fisher predicted signedweightMSE among128 symmetric8bit pairs(low0..127,high255-low), tieslowerlowcode. Uses onlynominal codes/spans, unitgain/zerooffset approximation. No device truth, measurements, textlabels or taskoutputs. Fixedbaselinecodes96/159. Allchoices frozenbefore any newcounts/modelscoring.',
        'device_model':'New independent profiles SeedSequence([2026116600,stage,seed]); standardnormal(2,N), trueoffset.2row0,truegainexp(.2row1-.02). Staticthroughbothprobes/inference. Syntheticdistribution, notphysicalmeasurements.',
        'calibration':'Exactly2048actualTHRML samples perprobe inbothmethods. Same smoothed-logit two-pointfit/gain guard[.25,4], inverseclipped/rounded8bit[-4,4]. Trueparams usedonlytorealizethe device andscore diagnosticerrors afterfitting, neverbychooser/estimator.',
        'keys':'Probe1160731foldseed,stage,probe0/1; inference1160730foldseed,stage,bank. Samekeysacrossmethodsforcommonrandomnumbers, newrootsandphysicalprofilesfromoldstudies. Statecountsinteger; allnewnativeprobes, no binomialreplacement.',
        'execution':'All12 firstthreeblockprojections sequentially replacedonactualcurrentstudentinputs. Onefreepbitpernormalizedsignedweight, zeroendpointclamps, same8bitcontrols/ranges asbefore. RestmodelFP32. Nativeoutputaccumulationbeforeoriginalnonlinearities.',
        'decision':'All240probewindows and10nativequarterforwardruns retained. PrimarypairedCE contrast weight_fisher-minus-fixed; descriptivecrossedseed/bankintervals10000resamplesseed2026116701. Fixedqualityrule PPLratio<=1.05, accuracydelta>=-.01; stricterCE97.5%upper<=log1.05 andaccuracy2.5%lower>=-.01. No task-drivenmethodselection or rescue iffailed.',
        'raw_evidence':'Preserve every native output, counts, commandcodes, per-token loss/prediction and SHA256 of every runtime input. Rawinputs heldinRAM duringdriver andone temporaryIPCfile. Independent audit regenerates allinputs fromoriginalmodel/prioroutputs, verifieshashes andnativepaths, thenremoves only regenerableauditcopies. This saves redundant FP32teacher/input storage withoutreducingmodel/windows/seeds.',
        'scope':'Newdisjointtextpositions/newdeviceprofiles, sameWikiText2corpus andoneDistilGPT2checkpoint,25.92%converted. UnitFisherdesignisapproximate; practicalsyntheticcounts-onlyestimator, notknownparameteroracle. Counts matched, notphysicalenergy/latency; actualdevice transfer/drift/correlation andfull-model performance unproved.',
        'source_sha256':{str(Path('research/thrml_spw')/n):sha(Path('research/thrml_spw')/n) for n in ['prepare_probe_placement.py','probe_placement_math.py']},'model_scores_computed':False,'probe_counts_collected':False}
    (OUT/'protocol.json').write_text(json.dumps(p,indent=2)+'\n');print(json.dumps({'prepared':True,'new_windows':64,'choices':[(r['stage'],r['chosen']['low_code'],r['chosen']['high_code']) for r in placements],'model_scores_computed':False,'probe_counts_collected':False},indent=2))


if __name__=='__main__':main()
