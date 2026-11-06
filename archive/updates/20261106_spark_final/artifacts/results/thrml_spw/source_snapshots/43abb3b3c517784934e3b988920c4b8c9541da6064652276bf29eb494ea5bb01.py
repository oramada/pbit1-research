"""Independent decode, variance and trajectory audit of strong spatial control."""
import os
os.environ['JAX_PLATFORMS']='cpu'
if hasattr(os,'sched_setaffinity'):os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[-2:])
from pathlib import Path
import hashlib,json
import numpy as np
from .prepare_v1 import OUT,sha
from research.thrml_spw.native import NativeProbabilityBank


def main():
 root=OUT/'native_coarse_fractional_v1';r=json.loads((root/'results.json').read_text());p=json.loads((root/'protocol.json').read_text());assert r['protocol_sha256']==sha(root/'protocol.json')
 with np.load(OUT/'native_head_v1/inputs.npz')as f:d={k:f[k].copy()for k in f.files}
 X=d['hidden'].astype(np.float64);W=d['original_weights'].astype(np.float64);programs={};summaries=[]
 for layout in ['storage_flat','transposed_matrix']:
  lo=d[layout+'_lower'].astype(np.float64);span=d[layout+'_span'].astype(np.float64);u=np.divide(W-lo,span,out=np.full(W.shape,.5),where=span!=0);u=np.clip(u,0,1);q=np.minimum(np.floor(u*8),7).astype(np.uint8);frac=8*u-q
  edge=1/(1+np.exp(8));prob=np.clip(frac,edge,1-edge);h=(np.log(prob)-np.log1p(-prob))/2;code=np.rint(np.clip((h+4)*4095/8,0,4095)).astype(np.uint16)
  with np.load(root/f'{layout}_program.npz')as f:np.testing.assert_array_equal(f['coarse_word'],q);np.testing.assert_array_equal(f['fraction_codes'],code)
  fields=code.astype(np.float32)*np.float32(8/4095)-np.float32(4);p4=1/(1+np.exp(-2*fields.astype(np.float64)))
  f1=d[layout+'_codes'].astype(np.float32)*np.float32(8/4095)-np.float32(4);p1=1/(1+np.exp(-2*f1.astype(np.float64)))
  bits=((q[None]>>np.arange(3)[:,None,None])&1).astype(bool);programs[layout]=(lo,span,p1,p4,q,bits)
  for kind,T in [('K1',32768),('K4_fractional',8192),('K4_fractional',32768)]:
   rows=[x for x in r['records']if x['layout']==layout and x['kind']==kind and x['T']==T];assert len(rows)==5 and sorted(x['seed']for x in rows)==p['seeds']
   mean=lo+span*(p1 if kind=='K1'else(q+p4)/8);variance=span**2*(p1*(1-p1)if kind=='K1'else p4*(1-p4)/64)/T
   theoretical=float(np.mean((X@(mean-W).T)**2)+np.mean(X**2@variance.T))
   for row in rows:
    file=Path(row['file']);assert sha(file)==row['file_sha256']
    with np.load(file)as f:counts=f['counts'].copy();estimate=f['estimate'].copy()
    if kind=='K1':assert counts.shape==W.shape;unit=counts.astype(np.float64)/T
    else:
     assert counts.shape==(4,*W.shape);np.testing.assert_array_equal(counts[:3],bits*T);unit=(q+counts[3].astype(np.float64)/T)/8
    np.testing.assert_array_equal(lo+span*unit,estimate);np.testing.assert_allclose(np.mean((X@(estimate-W).T)**2),row['output_MSE'],rtol=1e-13);np.testing.assert_allclose(theoretical,row['predicted_output_MSE'],rtol=1e-13)
    assert row['logical_pbits']==(1 if kind=='K1'else 4)*6144 and row['free_pbits']==6144 and row['clamped_pbits']==(0 if kind=='K1'else 18432)
    assert row['all_node_observations']==row['logical_pbits']*T and row['free_node_updates']==6144*T
   summaries.append({'layout':layout,'kind':kind,'T':T,'mean_MSE':float(np.mean([x['output_MSE']for x in rows])),'predicted_MSE':theoretical,'node_observations':rows[0]['all_node_observations'],'free_updates':rows[0]['free_node_updates']})
 replays=[]
 layout='transposed_matrix';lo,span,p1,p4,q,bits=programs[layout]
 for kind,T in [('K1',32768),('K4_fractional',8192),('K4_fractional',32768)]:
  bank=NativeProbabilityBank(p1 if kind=='K1'else np.concatenate([bits.astype(np.float64),p4[None]],axis=0));seed=p['seeds'][0];states=bank.sample(seed,T,1)[0]
  ref=next(x for x in r['records']if x['layout']==layout and x['kind']==kind and x['T']==T and x['seed']==seed)
  assert hashlib.sha256(np.ascontiguousarray(states).tobytes()).hexdigest()==ref['trajectory_sha256'];replays.append({'kind':kind,'T':T,'seed':seed,'trajectory_bitwise_equal':True});del states
 output={'status':'all30_cases_audited_and_three_native_trajectories_replayed','source_sha256':sha(__file__),'result_sha256':sha(root/'results.json'),'summary':summaries,'replays':replays,'scope':p['scope']}
 (OUT/'native_coarse_fractional_v1_audit.json').write_text(json.dumps(output,indent=2)+'\n');print(json.dumps(summaries,indent=2))
if __name__=='__main__':main()
