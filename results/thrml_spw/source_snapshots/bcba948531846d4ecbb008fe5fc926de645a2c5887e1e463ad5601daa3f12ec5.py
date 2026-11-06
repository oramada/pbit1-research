"""Audit completed conservative calibration arithmetic before confirmation finishes."""
import json
from pathlib import Path
import numpy as np
from .verify_quarter_conservative import independent_profile
from .verify_transformer_control import sha


def main():
    out=Path('results/thrml_spw/quarter_conservative');selection=json.loads((out/'selection.json').read_text());p=json.loads((out/'protocol.json').read_text())
    data=np.load(p['input_data_path']);base=np.load(out/'baseline_tokens.npz');count=0
    for record in selection['profiles']:
        t=record['samples'];raw=[np.load(out/'runs'/f'validation_T{t}_seed{s}'/'prefix_tokens.npz') for s in range(5)]
        independent,dc,da=independent_profile(np.stack([r['prefix12_nll'] for r in raw]),np.stack([r['prefix12_predictions'] for r in raw]),base['validation_FP32_nll'],base['validation_FP32_predictions'],data['validation_tokens'],t,20261111)
        for key,value in independent.items():
            if isinstance(value,bool):assert record[key]==value
            else:np.testing.assert_allclose(value,record[key],rtol=1e-12,atol=1e-15)
        saved=np.load(out/f'validation_T{t}_selection_draws.npz');np.testing.assert_allclose(saved['CE_difference'],dc,rtol=1e-12,atol=1e-15);np.testing.assert_allclose(saved['accuracy_difference'],da,rtol=1e-12,atol=1e-15);count+=20000
    passing=[r for r in selection['profiles'] if r['passes']];assert selection['gate_passed']==bool(passing)
    assert selection['selected_samples']==(passing[0]['samples'] if passing else p['sample_search'][-1])
    result={'passed':True,'selected_samples':selection['selected_samples'],'independently_recovered_bootstrap_values':count,
        'selection_sha256':sha(out/'selection.json'),'test_used_in_selection':False,'source_sha256':sha(__file__),
        'scope':'Independent calibration arithmetic and every saved selection draw from token evidence. Full raw-stage/model verification remains required after confirmation finishes.'}
    (out/'calibration_arithmetic_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
