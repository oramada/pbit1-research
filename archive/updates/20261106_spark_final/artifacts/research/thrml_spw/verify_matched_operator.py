"""Independent raw audit of the large matched spatial/temporal replica control."""
import csv,json
from pathlib import Path
import numpy as np
from .verify_transformer_control import sha


def main():
    out=Path('results/thrml_spw/matched_operator');p=json.loads((out/'protocol.json').read_text());report=json.loads((out/'REPORT.json').read_text())
    artifact=Path('results/thrml_spw/packed_quarter_controls/stage11.npz');source=Path('results/thrml_spw/transformer_quarter/runs/validation_T512_seed0')
    xp,rp=[source/f'stage11_{n}.npy' for n in ['inputs','conditional_reference']]
    assert sha(artifact)==p['artifact_sha256'] and sha(xp)==p['inputs_sha256'] and sha(rp)==p['reference_sha256']
    assert sha('research/thrml_spw/matched_operator.py')==p['source_sha256'];x,ref=np.load(xp),np.load(rp)
    with np.load(artifact,allow_pickle=False) as a:
        header=json.loads(a['header'].tobytes().decode());shape=tuple(header['shape']);n=int(np.prod(shape));bits=np.unpackbits(a['modes'],bitorder='little').reshape(-1,2)[:n]
        mode=bits[:,0]+2*bits[:,1];free=int(np.count_nonzero(mode==0));q=np.zeros(n,dtype=np.float64);q[mode==2]=1
        field=(-4+a['codes'].astype(np.float64)*(8/255)).astype(np.float32);q[mode==0]=np.exp(-np.logaddexp(0,-2*field.astype(np.float64)));q=q.reshape(shape)
        lower=np.repeat(a['lower'],128,0).astype(np.float64);span=np.repeat(a['span'],128,0).astype(np.float64);bias=a['bias'].copy()
    mean=x.astype(np.float64)@(lower+span*q)+bias;floor=float(np.mean((mean-ref)**2))
    variance=float(np.mean(np.mean(x.astype(np.float64)**2,axis=(0,1))@(span**2*q*(1-q)))/512)
    with (out/'metrics.csv').open() as f:rows=list(csv.DictReader(f))
    with (out/'summary.csv').open() as f:summaries=list(csv.DictReader(f))
    assert len(rows)==20 and {(r['condition'],int(r['seed'])) for r in rows}=={(c,s) for c in ['temporal','spatial'] for s in range(10)}
    outputs=np.load(out/'outputs.npz');mse_by_mode={};sampling_by_mode={}
    for r in rows:
        c=r['condition'];s=int(r['seed']);k,t=(1,512) if c=='temporal' else (4,128);y=outputs[f'{c}_seed{s}']
        assert (int(r['K']),int(r['T']))==(k,t);assert y.shape==ref.shape
        mse=float(np.mean((y.astype(np.float64)-ref)**2));centered=float(np.mean((y.astype(np.float64)-mean)**2))
        np.testing.assert_allclose(mse,float(r['conditional_mse']),rtol=1e-12);np.testing.assert_allclose(centered,float(r['centered_sampling_mse']),rtol=1e-12)
        np.testing.assert_allclose(floor,float(r['control_bias_mse']),rtol=1e-9)
        # Independent float64 squaring instead of the producer's float32 span square.
        np.testing.assert_allclose(variance,float(r['predicted_sampling_mse']),rtol=3e-7)
        assert int(r['unique_weights'])==n and int(r['logical_nodes'])==k*n and int(r['free_nodes'])==k*free
        assert int(r['free_node_evaluations'])==2*t*k*free and int(r['logical_weight_readouts'])==2*t*k*n
        assert int(r['nominal_binary_product_terms'])==16*65*t*k*n and int(r['actual_float32_field_bytes'])==4*k*free
        assert int(r['parallel_steps_per_bank'])==t and int(r['total_draws_per_unique_weight_per_bank'])==512
        mse_by_mode.setdefault(c,[]).append((s,mse));sampling_by_mode.setdefault(c,[]).append(centered)
    for c in mse_by_mode:mse_by_mode[c]=np.array([v for _,v in sorted(mse_by_mode[c])])
    for r in summaries:
        c=r['condition'];values=mse_by_mode[c];np.testing.assert_allclose(values.mean(),float(r['MSE_mean']),rtol=1e-12);np.testing.assert_allclose(1.96*values.std(ddof=1)/np.sqrt(10),float(r['MSE_mc95_margin']),rtol=1e-12)
    differences=mse_by_mode['spatial']-mse_by_mode['temporal']
    np.testing.assert_allclose(differences.mean(),report['spatial_minus_temporal_MSE_mean'],rtol=1e-12);np.testing.assert_allclose(1.96*differences.std(ddof=1)/np.sqrt(10),report['spatial_minus_temporal_MC95_margin'],rtol=1e-12)
    audit={'passed':True,'native_conditions':20,'seeds_per_mode':10,'matrix_weights':n,'matched_free_node_evaluations_per_condition':2*512*free,
        'source_sha256':sha(__file__),'scope':'Every saved output, MSE, analytic control floor/variance and resource counter independently recomputed. Replica control only, not a positional or official Extropic baseline, formal equivalence test or physical-system proof.'}
    (out/'evidence_audit.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps(audit,indent=2))
    temporal=next(r for r in summaries if r['condition']=='temporal');spatial=next(r for r in summaries if r['condition']=='spatial')
    d=float(differences.mean());m=float(1.96*differences.std(ddof=1)/np.sqrt(10))
    text=f'''# Native matched-resource large-operator replica control

The same2,359,296-weight programmed projection and previously scored old student
inputs were evaluated with one p-bit per weight for512 samples and four distinct
independent same-probability p-bits per weight for128 parallel samples. There are
ten fixed seeds per mode, with disjoint condition key streams. The sampler is the
official THRML Gibbs implementation; replica nodes are explicit distinct SpinNodes.
The controls are finite8-bit unary biases and shared group128 affine metadata.

This is a replica control, not positional K-bit encoding or an official Extropic
learned-weight reference. It does not establish that Extropic uses four p-bits
for each learned weight. See research/thrml_spw/REFERENCE_MAPPING.md.

| Alternative | Logical nodes | Parallel steps/bank | Free evaluations per condition | Mean output MSE | Approximate MC95 margin |
|---|---:|---:|---:|---:|---:|
| One temporal node/weight | {n:,} |512|{2*512*free:,}|{float(temporal['MSE_mean']):.7f}|{float(temporal['MSE_mc95_margin']):.7f}|
| Four replicas/weight | {4*n:,} |128|{2*128*4*free:,}|{float(spatial['MSE_mean']):.7f}|{float(spatial['MSE_mc95_margin']):.7f}|

The spatial-minus-temporal MSE difference is {d:.7g}, with approximate ten-seed
normal interval[{d-m:.7g},{d+m:.7g}]. Covering zero would be consistency with the
same-budget prediction, not a formal statistical equivalence proof. The analytic
sampling MSE is {variance:.7g} for both, with a common control-bias floor {floor:.7g}.
Both modes have the same512 total observations per unique weight, and therefore
the same Bernoulli variance law under independent nodes and draws.

Logical read totals and nominal binary-product terms also match. Four replicas
have four times the logical device population and read width, one-quarter the
parallel step depth, and four times the simulator's float32 field storage. Shared
physical bias fanout is possible in principle but is not modeled or measured.
Temporal sampling does not provide free precision, and equal draw counts do not
show an energy win. The reported steps are not measured physical-chip latency.
External floating arithmetic and separate replica accumulators also cost resources.

Every output and counter was independently audited; one tiny CPU/CUDA same-path
check additionally compares reconstruction against the explicit replica trajectories.
This is not a full neural-network quality comparison. The ongoing conservative
confirmation uses its unchanged canonical representation and does not read these
results or old inputs. Raw evidence is under matched_operator/.
'''
    for a,b in {'same2,359':'same 2,359','for512':'for 512','for128':'for 128','finite8':'finite 8','interval[':'interval [','floor{':'floor {','same512':'same 512'}.items():text=text.replace(a,b)
    Path('results/thrml_spw/MATCHED_OPERATOR_REPORT.md').write_text(text)

if __name__=='__main__':main()
