"""Exercise actual producer checkpoint writer with relative/absolute roots."""
from pathlib import Path
import json,hashlib,os,tempfile
import numpy as np
from .holdout_native_case_v4 import case_directory,commit_closed_stage,token_nll
from .stage_checkpoint_v1 import StageStore,sha
from .lossless_logits import write_logits,read_logits
from .verify_holdout_native_case_v4 import independent_losses

def main():
    out=Path('results/thrml_spw/broad_verification_20261105/case_checkpoint_v4_preflight')
    out.mkdir(exist_ok=False)
    rng=np.random.default_rng(202611904)
    logits=rng.normal(size=(2,5,7)).astype(np.float32)
    ids=rng.integers(0,7,(2,5),dtype=np.int64)
    losses=token_nll(logits,ids)
    np.testing.assert_allclose(losses,independent_losses(logits,ids),rtol=1e-12,atol=1e-12)
    records=[]
    for supplied in [out/'relative', (out/'absolute').resolve()]:
        directory=case_directory(supplied,202611150)
        assert directory.is_absolute()
        store=StageStore(directory,{'fixture':'producer_path_and_closed_stage','seed':202611150})
        attempt=store.new_attempt();path=attempt/'batch0_BF16.npz'
        codec=write_logits(path,logits)
        item={'batch':0,'stage':'BF16','path':str(path),'sha256':sha(path),**codec}
        metric={'batch_index':0,'stage':'BF16','mean_NLL_FP64':float(losses.mean()),'token_NLL':losses.tolist()}
        commit_closed_stage(store,0,'BF16',path,item,metric,['synthetic_module'],[],None)
        resumed=StageStore(directory,{'fixture':'producer_path_and_closed_stage','seed':202611150})
        closed=resumed.load(0,'BF16')
        assert closed['metrics']==metric and closed['output_item']==item
        assert closed['output']['path']==str(path.relative_to(directory))
        np.testing.assert_array_equal(read_logits(directory/closed['output']['path']).view(np.uint32),logits.view(np.uint32))
        records.append({'supplied_root_was_absolute':supplied.is_absolute(),'output_sha256':sha(path),'resume_record_identical':True})
    report={'status':'actual_v4_producer_writer_relative_absolute_roots_loss_codec_and_resume_passed',
        'cases':records,'sources_sha256':{n:sha(Path(__file__).with_name(n)) for n in [
            'preflight_case_checkpoint_v4.py','holdout_native_case_v4.py','verify_holdout_native_case_v4.py','stage_checkpoint_v1.py','lossless_logits.py']},
        'scope':'Small synthetic float32 observations and actual producer checkpoint helper; no real text, neural forward or new MonteCarlo quality evidence.'}
    (out/'results.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))

if __name__=='__main__':main()
