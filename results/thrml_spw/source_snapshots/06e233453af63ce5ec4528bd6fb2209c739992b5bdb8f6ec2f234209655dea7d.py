"""Close bounded convolution witness and quantify an explicit padding control."""
from pathlib import Path
import hashlib,json,platform,sys,datetime
from importlib.metadata import version
import numpy as np
ROOT=Path('results/thrml_spw');OUT=ROOT/'convolution_native'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    p=json.loads((OUT/'protocol.json').read_text());m=json.loads((OUT/'manifest.json').read_text());a=json.loads((OUT/'evidence_audit.json').read_text());assert a['passed'] and len(m['rows'])==12
    for f,h in p['source_sha256'].items():assert sha(f)==h
    d=np.load(OUT/'inputs.npz');lo,span=p['activation_lower'],p['activation_span'];zero_code=round(-lo/span*255);quant_zero=lo+span*zero_code/255
    rows=[]
    for row in m['rows']:
        assert sha(row['path'])==row['sha256'];raw=np.load(row['path']);n=raw['codes'].shape[0];missing_counts=raw['column'][:,None,:].astype(float)-raw['gated'][:,n:].astype(float);missing_slots=(1-raw['mask']).sum(1,dtype=np.uint32)
        # Counterfactual convolution pads with rounded quantized zero rather than exact zero.
        # Both the range offset and sampled correction then leak into padding.
        delta=quant_zero*((d['center']-d['radius'])[None,None,:]*missing_slots[None,:,None]+2*d['radius'][None,None,:]*missing_counts/row['T'])
        assert np.isfinite(delta).all()
        maximum=float(abs(delta).max());nonzero=int(np.count_nonzero(abs(delta)>1e-12))
        if row['geometry']==0:assert maximum==0
        else:assert maximum>1e-6 and nonzero>0
        rows.append({'case':row['name'],'max_abs_padding_error_if_rounded_zero_used':maximum,'nonzero_output_coordinates':nonzero})
    (OUT/'padding_negative_control.json').write_text(json.dumps({'scope':'Post-witness diagnostic, same retained trajectories, no new samples or gate. Algebraic effect of replacing exact zero padding by rounded affine-grid zero.','zero_code':zero_code,'quantized_zero':quant_zero,'rows':rows},indent=2)+'\n')
    environment={'python':sys.version,'platform':platform.platform(),'packages':{n:version(n) for n in ['thrml','jax','jaxlib','numpy','equinox']},'backend':'cpu','x64_enabled':False,'XLA_FLAGS':'--xla_cpu_multi_thread_eigen=false','key_root':p['key_root'],'vendor_commit':p['vendor_commit']}
    (OUT/'environment.json').write_text(json.dumps(environment,indent=2)+'\n')
    max_pad=max(r['max_abs_padding_error_if_rounded_zero_used'] for r in rows)
    text=f'''# Native signed-range convolution operator witness

Extropic's pinned official THRML sampler executes one free p-bit per normalized kernel weight. A signed weight is mapped by w=(c-r)+2*r*p, with per-output-channel center/radius. Native integer observers accumulate activation-code gates and separate validity-mask gates using the same sampled weights. There are no activation p-bits, per-weight operational reconstruction counters or floating neural multiplies inside the observer. The sampler still uses floating probability arithmetic, and final range correction/scaling/bias use external F64 arithmetic.

## Executed coverage

All12 predeclared cases cover valid convolution, padding1, stride2 and dilation2/padding2 at T1/7/64. Each has two independent sampling keys, shared across two synthetic input images and every spatial patch. Kernels have3 output channels,2 input channels and3x3 support:54 weight p-bits per chain. Signed weights, clipped inputs, negative activations, finite8-bit weight-field commands±4 and exact zero padding are included. This is a bounded implementation witness, not a five-seed Monte Carlo quality study or a trained CNN result.

Public raw-state sampling independently reconstructs all{a['counter_entries_exact']} stored integer gate/column counters exactly. Separate nested spatial loops, without the production im2col implementation, reconstruct every convolution output with maximum absolute error {a['max_output_abs_error']:.17g}. Production executes{a['production_updates']} scheduled node updates; independent verification executes another{a['verification_updates']}. Same-key verification is not a new independent replication. Weight draws are reused across positions/images; those outputs are correlated rather than independent samples.

## Padding and accounting

For each patch, let M=sum_t,i valid_i*Z_io(t) and G=sum_t,i code_i*valid_i*Z_io(t). The probabilistic contribution is lower*M/T+(span/255)*G/T. The signed-range offset multiplies the sum of valid quantized activations. An explicit mask prevents padded pixels from acquiring rounding error or the affine lower offset. The shared weight-column observer remains a diagnostic counter, separate from patch-specific validity sums.

Here lower=-0.7 and span=2.3 make the rounded code for zero {zero_code}, reconstructing {quant_zero:.12g}. A post-witness negative control derives the effect of using that rounded value for padding from the same retained counters: maximum spurious output error {max_pad:.12g}; valid convolution has exactly zero padding-control difference. All padded geometries show nonzero leakage. This diagnostic changes no protocol or acceptance tolerance and draws no additional states.

Deterministic activation codes/masks, per-output center/radius, bias/field commands, patch counters and external F64 readout all remain costs. The observer retains two output arrays per patch plus shared diagnostic column counts, using native int32. Spatial parallelism, im2col/software temporaries, physical multipliers/gates, controller state, area/energy and readout latency are not measured. Logical one-p-bit kernel storage does not establish one total bit of parameter information or a hardware advantage.

## Next scope

This closes the convolution operator prerequisite. Native full CNN conversion, all-layer nonlinear ordering, validation-only range calibration, five independently trained checkpoints, full held-out evaluation and matched conventional controls remain required. The current full MNIST classifier and queued static-cache/integer-activation cohorts keep their original protocols. The long-horizon research goal remains active.

[Protocol](convolution_native/protocol.json) · [Production](convolution_native/manifest.json) · [Independent audit](convolution_native/evidence_audit.json) · [Padding control](convolution_native/padding_negative_control.json) · [Environment](convolution_native/environment.json)
'''
    report=ROOT/'CONVOLUTION_NATIVE_REPORT.md';report.write_text(text)
    paths=[q for q in OUT.rglob('*') if q.is_file() and '__pycache__' not in q.parts and q.name!='completion.json' and not q.name.endswith('.tmp')]+[report]
    completion={'status':'bounded_native_convolution_audited_goal_active','recorded_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'artifact_sha256':{str(f):sha(f) for f in sorted(paths)},'global_index':'Stale while unrelated classifier/followup writers are live; not refreshed here.'}
    (OUT/'completion.json').write_text(json.dumps(completion,indent=2)+'\n');print(json.dumps({'status':completion['status'],'artifacts':len(paths),'maximum_padding_control_error':max_pad}))
if __name__=='__main__':main()
