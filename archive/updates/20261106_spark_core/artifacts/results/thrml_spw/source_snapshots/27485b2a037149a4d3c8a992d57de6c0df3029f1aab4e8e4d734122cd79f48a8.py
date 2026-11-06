"""Recompute saved full-operator metrics and command means independently."""
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    root=Path('results/thrml_spw/lfm3_1p7b');out=root/'full_operator'
    result=json.loads((out/'results.json').read_text())
    protocol=json.loads((out/'protocol.json').read_text())
    assert sha(root/'full_operator_export.npz')==protocol['export_sha256']
    assert sha(out/'outputs.npz')==result['outputs_sha256']
    assert len(result['cases'])==6
    data=np.load(root/'full_operator_export.npz');raw=np.load(out/'outputs.npz')
    metrics=0;coordinates=0
    for kind in ['linear','conv']:
        x=data[kind+'_inputs'].astype(np.float64);w=data[kind+'_weight'].astype(np.float64)
        step=np.max(np.abs(x))/127 if np.any(x) else 1.0
        q=np.rint(x/step).clip(-127,127)*step
        codes=raw[kind+'_field_codes'];assert codes.shape==w.shape and codes.dtype==np.uint8
        # Reproduce actual float32 command decode, then evaluate its probability in float64.
        h=np.asarray(codes,dtype=np.float32)*np.float32(8/255)-np.float32(4)
        prob=np.exp(-np.logaddexp(0,-2*h.astype(np.float64)))
        lo=np.min(w,axis=0);width=np.ptp(w,axis=0)
        meanweight=lo+width*prob
        def dot(a,b):return np.sum(a*b[None,:,:],axis=1) if kind=='conv' else np.matmul(a,b)
        ref=dot(x,w);act=dot(q,w);mean=dot(q,meanweight)
        np.testing.assert_allclose(mean,raw[kind+'_nominal_mean'],rtol=2e-12,atol=2e-12)
        coordinates+=w.size
        for t in [1,16,64]:
            row=next(r for r in result['cases'] if r['operator']==kind and r['samples']==t)
            y=raw[f'{kind}_T{t}'];assert np.isfinite(y).all() and y.shape==ref.shape
            for key,a in [('output_MSE',y),('ACT8_floor_MSE',act),('nominal_control_floor_MSE',mean)]:
                np.testing.assert_allclose(row[key],np.mean((a-ref)**2),rtol=2e-12,atol=2e-14);metrics+=1
            assert row['weights_executed']==w.size and row['tiles']==w.shape[1]//64
            assert row['retained_node_evaluations_per_pass']==t*w.size
            assert row['warmup_node_evaluations_per_pass']==w.size
            assert row['all_integer_public_replays_exact']
    audit={'status':'all_saved_metrics_and_commands_verified','source_sha256':sha(Path(__file__)),
           'results_sha256':sha(out/'results.json'),'metrics_recomputed':metrics,
           'weight_command_coordinates':coordinates,'operator_cases':6,
           'scope':'Independent saved metric/command/readout audit. Exact public-state trajectory comparison executed in producer; states not retained. One seed and one prompt; no language quality inference.'}
    (out/'independent_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit))


if __name__=='__main__':main()
