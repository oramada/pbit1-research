"""Freeze new quarter-model nonideal study before any new text scoring."""
from pathlib import Path
import json
import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from huggingface_hub.constants import HF_HUB_CACHE
from transformers import AutoTokenizer
from .transformer_export import MODEL_REVISION,DATASET_REVISION
from .verify_transformer_control import sha
from .calibrated_transformer_controls import export_controls


def main():
    root=Path('results/thrml_spw');out=root/'calibrated_transformer';out.mkdir(exist_ok=True);(out/'controls').mkdir(exist_ok=True)
    pp=out/'protocol.json'
    if pp.exists():raise RuntimeError('Immutable study already prepared; verify/reuse it')
    exclusions=[]
    for name,key in [('data/thrml_spw/transformer_projection/export.npz','test_token_offsets'),('data/thrml_spw/transformer_confirmation/tokens.npz','token_offsets'),('data/thrml_spw/quarter_confirmation/tokens.npz','token_offsets'),('data/thrml_spw/quarter_conservative/tokens.npz','test_token_offsets')]:
        exclusions.append({'path':name,'key':key,'sha256':sha(name)})
    used=np.concatenate([np.load(e['path'])[e['key']] for e in exclusions])
    source=Path(hf_hub_download('Salesforce/wikitext','wikitext-2-raw-v1/test-00000-of-00001.parquet',repo_type='dataset',revision=DATASET_REVISION,local_files_only=True))
    texts=pq.read_table(source,columns=['text']).column('text').to_pylist();tokenizer=AutoTokenizer.from_pretrained(Path(HF_HUB_CACHE)/f'models--distilbert--distilgpt2/snapshots/{MODEL_REVISION}',local_files_only=True)
    ids=np.asarray(tokenizer('\n'.join(t for t in texts if t and t.strip()),add_special_tokens=False)['input_ids'],dtype=np.int64)
    legacy=np.sort(np.random.default_rng(17).choice(len(ids)-64,size=4,replace=False));used=np.concatenate([used,legacy])
    eligible=np.arange(len(ids)//65)*65;eligible=eligible[~np.any((eligible[:,None]<used[None]+65)&(used[None]<eligible[:,None]+65),axis=1)]
    offsets=np.sort(np.random.default_rng(2026114700).choice(eligible,64,replace=False));tokens=np.stack([ids[i:i+65] for i in offsets]);np.savez_compressed(out/'tokens.npz',tokens=tokens,token_offsets=offsets)
    master=root/'transformer_quarter/weights.npz';weights=np.load(master);controls=[]
    for stage in range(12):
        cp=out/'controls'/f'stage{stage}.npz';export_controls(cp,weights[f'weight_{stage}'],weights[f'bias_{stage}'])
        controls.append({'stage':stage,'path':str(cp),'sha256':sha(cp),'shape':list(weights[f'weight_{stage}'].shape),'weights':int(weights[f'weight_{stage}'].size)})
    protocol={'schema':'spw-all-free-calibrated-quarter-transformer-v1','revision':MODEL_REVISION,'dataset_revision':DATASET_REVISION,'source_parquet_path':str(source),'source_parquet_sha256':sha(source),
        'excluded_archives':exclusions,'legacy_excluded_offsets':legacy.tolist(),'token_selection_seed':2026114700,'tokens_sha256':sha(out/'tokens.npz'),'windows':64,'sequence_tokens':65,'targets':4096,
        'targets_in_forward_order':json.loads((root/'transformer_quarter/protocol.json').read_text())['targets_in_forward_order'],
        'master_weights_path':str(master),'master_weights_sha256':sha(master),'control_artifacts':controls,'converted_weights':21233664,'model_unique_parameters':81912576,
        'conditions':['ideal','uncalibrated','cal4096'],'joint_device_sampling_seeds':list(range(5)),'samples_per_operator':1024,'contexts_per_bank':8,'controller_bits':8,'field_bound':4.,'input_block_size':128,
        'representation':'Signedmatrix W=lower+span*p withper-outputinput-block128 FP32ranges. OriginalFP32min/maxnormalizedp computedF32 thenclippedF64 tosigmoid(-8),1-sigmoid(-8). Allweightnodesfreelysampled,noexactclamps. Nominalfieldfloat32(logit(p)/2) thennearest8-bitcode[-4,4]. Onepbitpernormalizedprobability, notdirectp=rawsignedweight.',
        'device_model':'Static independent gainexp(.2*z1-.02),offset.2*z0 perweight. RNGSeedSequence([2026114600,stage,seed]) standardnormal(2,N). Fixedacrossprobes,conditionsandallinputbanks. No drift, temporal/cross-devicecorrelation orreadoutnoise; distributions are synthetic, not measured hardware.',
        'calibration':'Two actual nativeTHRML probes of2048samples at8bitcodes96/159 perfreeweight; counts-onlysmoothed-logit gain/offsetfit withguard[.25,4], corrected8bitboundedcommands. No trueparams,textinputs,labelsoroutputmetricsenterfit. Calcounts sharedacrossallthreeconditionsoneprofile, countedseparately.',
        'keys':'Inference1140730foldseed,stage,bank; identicalkey streamsacrossconditions butstatesdependonfields. Probe1140731foldseed,stage,probe0/1. Newrootsanddeviceprofilesdisjointfrompreviousoperator/neuralstudies.',
        'execution':'All12 projections in first3blocks replacedsequentially onactualcurrentstudentinputs. Regenerateandcheck earlierreplacementinputs on everyprefix. NativeTHRML linear-outputsum beforeoriginalTransformer nonlinearities; sharedweightsper8contextbank. Restofmodel deterministicFP32. Softwareexternalfloatingarithmetic, notphysicalTSUoperators.',
        'decision':'All15fullquarterforwardruns and120probe windows; no selection, sample search or test-driven extension. T1024 fixed frompriorindependentidealquarterwork; P4096 fixedfrompriorcalibrationwork. Primarycomparisoncal4096 versusuncalibrated. Idealallfreecontrol andFP32/affine4/affine8/programmed-mean baselines retained.',
        'quality_rule':'ReportfixedmeanrelativePPL<=1.05 andtokenaccuracyloss<=.01 versusFP32. Also reportpredeclaredcrossedseed/bankbootstrap97.5%-CE<=log1.05 and2.5%-accuracy>=-.01,10000resamplesseed2026114701. Heuristicexchangeablebanks, notformalcoverage. Retainallfailureswithoutrescue.',
        'scope':'SamepinnedWikiText2corpus,newnonoverlappingpositions,64shortcontexts4096targets. Firstthreeblocksonly25.92%modelweights, notfullmodel,longcontextorcorpusgenerality. Fivejointdevice+MCseeds, notfivepretrainedmodels. Calibration/control/rangemetadataandreadoutcostsareadditional; nohardwareadvantageclaim.',
        'source_sha256':{str(Path('research/thrml_spw')/n):sha(Path('research/thrml_spw')/n) for n in ['prepare_calibrated_transformer.py','calibrated_transformer_native.py','calibrated_transformer_controls.py']},'model_scores_computed':False}
    assert sum(c['weights'] for c in controls)==21233664
    pp.write_text(json.dumps(protocol,indent=2)+'\n');print(json.dumps({'prepared':True,'windows':64,'targets':4096,'controls':12,'converted_weights':21233664,'model_scores_computed':False},indent=2))


if __name__=='__main__':main()
