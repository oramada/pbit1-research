"""Independent all-case integer identity from public-replayed parent count traces."""
import json
from pathlib import Path
import numpy as np
from results.thrml_spw.integer_activation_native.cohort_common import OUT,PARENT,sha,guard,parent_closed,atomic_json

def main():
    p=json.loads((OUT/'protocol.json').read_text());prep=guard(p);parent_closed();m=json.loads((OUT/'manifest.json').read_text());assert m['complete'] and m['protocol_sha256']==sha(OUT/'protocol.json');pa=json.loads((PARENT/'native_audit.json').read_text());verified={r['name']:r for r in pa['rows']};x=np.load(prep['data_path'])['test_inputs'].astype(float).reshape(1250,8,64);rows=[];entries=0
    for row in m['rows']:
        assert sha(row['path'])==row['sha256'] and sha(row['parent_path'])==row['parent_sha256'];assert verified[row['parent_name']]['raw_sha256']==row['parent_sha256'];raw=np.load(row['path']);parent=np.load(row['parent_path']);model=np.load(prep['models'][row['model']]['controls_path']);h=x.copy();t=row['T'];clip=0;count=0
        for layer,(ni,no) in enumerate([(64,32),(32,10)]):
            low=float(model['input_lower'] if layer==0 else model['hidden_lower']);span=float(model['input_span'] if layer==0 else model['hidden_span']);a=np.round(np.minimum(1,np.maximum(0,(h-low)/span))*255).astype(np.uint8)
            if layer==1:assert raw['hidden_codes'].dtype==np.uint8;np.testing.assert_array_equal(a,raw['hidden_codes'])
            wc=parent[f'weight_{layer}'].astype(np.uint32);g=np.einsum('cbi,cio->cbo',a.astype(np.uint32),wc,dtype=np.uint32,optimize=False);c=wc.sum(1,dtype=np.uint32);assert raw[f'gated_{layer}'].dtype==raw[f'column_{layer}'].dtype==np.uint32;np.testing.assert_array_equal(g,raw[f'gated_{layer}']);np.testing.assert_array_equal(c,raw[f'column_{layer}']);count+=g.size+c.size
            y=low*c[:,None,:].astype(float)/t+(span/255)*g.astype(float)/t+model[f'bias_{layer}'];xr=low+span*a.astype(float)/255;reference=np.matmul(xr,wc.astype(float)/t)+model[f'bias_{layer}'];np.testing.assert_allclose(y,reference,rtol=2e-12,atol=2e-11);h=np.maximum(y,0) if layer==0 else y
            if layer==0:clip=int(np.count_nonzero((h<float(model['hidden_lower']))|(h>float(model['hidden_lower']+model['hidden_span']))))
        np.testing.assert_array_equal(h.reshape(10000,10),raw['logits']);assert clip==row['hidden_clipped_coordinates'];assert row['input_outside_affine_range']==np.count_nonzero(((x-prep['input_lower'])/prep['input_span']<0)|((x-prep['input_lower'])/prep['input_span']>1));entries+=count;rows.append({'name':row['name'],'raw_sha256':row['sha256'],'parent_public_native_case':row['parent_name'],'integer_counter_entries_exact':count,'all_derived_hidden_codes_and_logits_exact':True});atomic_json(OUT/'audit_progress.json',{'rows':rows,'counter_entries_exact':entries});print(json.dumps({'audited_cases':len(rows),'last':row['name']}),flush=True)
    assert len(rows)==150 and entries==70875000;guard(p);atomic_json(OUT/'evidence_audit.json',{'passed':True,'rows':rows,'cases':150,'integer_counter_entries_exact':entries,'all_full_causal_outputs_checked':True,'parent_public_native_audit_sha256':sha(PARENT/'native_audit.json'),'producer_manifest_sha256':sha(OUT/'manifest.json'),'new_public_raw_updates':0,'scope':'Complete new native output-counter trajectory/causal checks via already public-replayed same-key parent per-weight sufficient statistics. No new independent MC draws or new raw-state replay for this full cohort; separate bounded public raw-state witness retained.'})
if __name__=='__main__':main()
