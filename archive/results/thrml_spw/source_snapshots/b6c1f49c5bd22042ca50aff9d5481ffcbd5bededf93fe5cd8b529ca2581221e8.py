"""Early complete numerical-control audit; no labels, metrics or native draws."""
from pathlib import Path
import hashlib,json,struct,datetime
import numpy as np
from results.thrml_spw.fashion_cnn_native.cohort.common import ROOT,OUT,TRAIN,CONTROLS,sha,guard,numpy_control
from results.thrml_spw.fashion_cnn_native.cohort.verify import weights_independent
DEST=ROOT/'control_review_evidence'
def main():
    DEST.mkdir(exist_ok=True);assert not (DEST/'audit.json').exists(),'No duplicate complete control audit'
    p=json.loads((OUT/'protocol.json').read_text());guard(p);control=json.loads((OUT/'controls.json').read_text());assert control['complete'] and control['protocol_sha256']==sha(OUT/'protocol.json') and control['test_data_sha256']==sha(OUT/'test_data.npz') and control['control_logits_sha256']==sha(OUT/'control_logits.npz')
    sources=dict(p['source_sha256']);sources[str(Path(__file__))]=sha(__file__);protocol={'recorded_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_sha256':sources,'parent_protocol_sha256':sha(OUT/'protocol.json'),'parent_controls_sha256':sha(OUT/'controls.json'),'scope':'All50 predetermined conventional-control paths/full10000inputimages audited early while native production runs. No test labels, accuracy selection, new stochastic draws or source/protocol changes. These checks are the already frozen final auditor conditions, not relaxed tolerances.'};(DEST/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    data=np.load(OUT/'test_data.npz');x=data['inputs'];raw=Path('data/torchvision/FashionMNIST/raw/t10k-images-idx3-ubyte').read_bytes();assert struct.unpack('>IIII',raw[:16])==(2051,10000,28,28);pixels=np.frombuffer(raw,np.uint8,offset=16).reshape(10000,28,28).astype(np.float32)/np.float32(255);reference=((pixels[:,0::2,0::2]+pixels[:,0::2,1::2]+pixels[:,1::2,0::2]+pixels[:,1::2,1::2])/np.float32(4))[:,None];np.testing.assert_array_equal(reference,x)
    models=json.loads((TRAIN/'manifest.json').read_text())['rows'];saved=np.load(OUT/'control_logits.npz');rows=[];failures=[]
    for mi,row in enumerate(models):
        model=np.load(row['path']);assert sha(row['path'])==row['sha256']
        for ci,name in enumerate(CONTROLS):
            weights=weights_independent(model,name);ref=np.concatenate([numpy_control(x[first:first+128],model,weights,name.startswith('ACT8')) for first in range(0,10000,128)]);actual=saved[f'c{ci}_model{mi}'];tol=(p['control_tolerances']['F32_rtol'],p['control_tolerances']['F32_atol']) if name=='FP32_F32' else (p['control_tolerances']['F64_rtol'],p['control_tolerances']['F64_atol']);record={'model':mi,'control':name,'max_abs_error':float(np.max(abs(ref-actual))),'prediction_mismatches':int(np.count_nonzero(ref.argmax(1)!=actual.argmax(1))),'rtol':tol[0],'atol':tol[1]}
            try:np.testing.assert_allclose(ref,actual,rtol=tol[0],atol=tol[1]);np.testing.assert_array_equal(ref.argmax(1),actual.argmax(1));record['passed']=True
            except AssertionError as e:record.update(passed=False,failure=str(e)[:1400]);failures.append(record)
            rows.append(record);print(json.dumps(record),flush=True)
    guard(p)
    for file,h in sources.items():assert sha(file)==h,file
    result={'passed':not failures,'rows':rows,'failed_paths':failures,'controls_checked':50,'input_coordinates_exact':1960000,'logit_coordinates_checked':5000000,'predictions_compared':500000,'new_native_updates':0,'protocol_sha256':sha(DEST/'protocol.json'),'test_data_sha256':sha(OUT/'test_data.npz'),'control_logits_sha256':sha(OUT/'control_logits.npz'),'numpy_version':np.__version__,'scope':protocol['scope']};(DEST/'audit.json').write_text(json.dumps(result,indent=2)+'\n');raise SystemExit(0 if result['passed'] else 1)
if __name__=='__main__':main()
