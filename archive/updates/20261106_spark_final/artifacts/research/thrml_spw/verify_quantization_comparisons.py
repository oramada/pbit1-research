"""Recompute every diagnostic resample independently of the producer helper."""
import json
from pathlib import Path
import numpy as np
from .fine_range_common import sha,atomic_json
OUT=Path('results/thrml_spw/fine_range_quantization_controls')
def main():
    p=json.loads((OUT/'protocol.json').read_text());m=json.loads((OUT/'comparison_manifest.json').read_text());assert json.loads((OUT/'evidence_audit.json').read_text())['passed'] and sha(OUT/'paired_draws.npz')==m['draws_sha256'] and sha(OUT/'paired_comparisons.json')==m['paired_sha256'];rows=json.loads((OUT/'paired_comparisons.json').read_text());draws=np.load(OUT/'paired_draws.npz');det=np.load(OUT/'raw_tokens.npz');truth=np.load(p['source_tokens'])['test_tokens'][:,1:];rng=np.random.default_rng(2026118200);si=rng.integers(0,5,(10000,5));bi=rng.integers(0,4,(10000,4));count=0
    for row in rows:
        rr=[np.load(Path(p['source_cohort'])/'runs'/f'{row["native"]}_seed{s}/prefix_tokens.npz') for s in range(5)];loss=np.stack([r['prefix24_nll'].astype(float).mean(-1) for r in rr]);acc=np.stack([(r['prefix24_predictions']==truth).mean(-1) for r in rr]);ref=det[row['control']+'_nll'].astype(float).mean(-1);refac=(det[row['control']+'_predictions']==truth).mean(-1);key=row['native']+'_minus_'+row['control']
        for label,delta in [('CE',loss-ref),('accuracy',acc-refac)]:
            blocks=delta.reshape(5,4,8).mean(-1);expected=np.array([blocks[si[k]][:,bi[k]].mean() for k in range(10000)]);np.testing.assert_allclose(expected,draws[key+'_'+label],rtol=1e-12,atol=1e-12);np.testing.assert_allclose(np.quantile(expected,.025),row[label+'_sensitivity_low'],rtol=1e-12,atol=1e-12);np.testing.assert_allclose(np.quantile(expected,.975),row[label+'_sensitivity_high'],rtol=1e-12,atol=1e-12);count+=len(expected)
        for k,v in {'CE_native_minus_control':(loss-ref).mean(),'perplexity_native_over_control':np.exp((loss-ref).mean()),'accuracy_native_minus_control':(acc-refac).mean()}.items():np.testing.assert_allclose(v,row[k],rtol=1e-12,atol=1e-12)
    assert count==440000 and len(rows)==22;r={'passed':True,'independent_resampling_values_checked':count,'comparisons_checked':22,'source_sha256':sha(__file__),'scope':'Diagnostic consumed-text paired comparisons only, not new generalization confirmation or physical performance.'};atomic_json(OUT/'comparison_audit.json',r);print(json.dumps(r,indent=2))
if __name__=='__main__':main()
