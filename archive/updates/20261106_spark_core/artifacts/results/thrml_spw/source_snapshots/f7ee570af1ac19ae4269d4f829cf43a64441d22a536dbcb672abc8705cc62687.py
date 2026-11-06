"""Strong K4 coarse-plus-fractional control using official THRML samplers."""
import os
os.environ['JAX_PLATFORMS']='cpu'
if hasattr(os,'sched_setaffinity'):os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[-2:])
from pathlib import Path
import hashlib,json,time
import numpy as np
from .prepare_v1 import OUT,sha
from research.thrml_spw.native import NativeProbabilityBank


def fractional_codes(f):
 edge=1/(1+np.exp(8.));p=np.clip(f,edge,1-edge);half=.5*(np.log(p)-np.log1p(-p));codes=np.rint(np.clip((half+4)*(4095/8),0,4095)).astype(np.uint16)
 field=codes.astype(np.float32)*np.float32(8/4095)-np.float32(4);return codes,1/(1+np.exp(-2*field.astype(np.float64)))


def main():
 root=OUT/'native_coarse_fractional_v1';root.mkdir(exist_ok=False);begin=time.monotonic()
 protocol={'source_sha256':sha(__file__),'native_source_sha256':sha('research/thrml_spw/native.py'),'inputs_sha256':sha(OUT/'native_head_v1/inputs.npz'),'seeds':list(range(2026112601,2026112606)),
 'K1_T':32768,'K4_T':[8192,32768],'coarse_bits':3,'K4_definition':'For normalizedu in[0,1], q=min(floor(8u),7), f=8u-q. Storeq inthreeclampedbinarynodes andf inonefreepbit withsame12bitfieldgrid asK1. Decodeu=(q+z)/8. Fourphysicalelements;onlyonefree stochastic element. Freefractionendpoints usefinitebias notexactclamping.',
 'ideal_bits_per_parameter_without_ranges':{'K1_controls_and_state':13,'K4_controls_coarse_commands_and_states':19},
 'scope':'Real8x768GPT2headbank and8fixedhiddenvectors, officialTHRML. Stronginternalcoarse/fractionalbaseline, notExtropicproductionencoding. K4T8192matchesall-nodeobservations; K4T32768matchesfree-nodeupdates. Clampednodesdo notswitchrandomly: countthemdifferently. No fullmodelquality, hardwareenergy, or universal dominance claim.'}
 (root/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
 with np.load(OUT/'native_head_v1/inputs.npz')as f:data={k:f[k].copy()for k in f.files}
 X=data['hidden'].astype(np.float64);W=data['original_weights'].astype(np.float64);records=[]
 try:
  for layout in ['storage_flat','transposed_matrix']:
   lo=data[layout+'_lower'].astype(np.float64);span=data[layout+'_span'].astype(np.float64);unit=np.divide(W-lo,span,out=np.full(W.shape,.5),where=span!=0);assert(unit>=-1e-12).all()and(unit<=1+1e-12).all();unit=np.clip(unit,0,1)
   field=data[layout+'_codes'].astype(np.float32)*np.float32(8/4095)-np.float32(4);p1=1/(1+np.exp(-2*field.astype(np.float64)))
   q=np.minimum(np.floor(8*unit),7).astype(np.uint8);fraction=8*unit-q;codes,p4=fractional_codes(fraction);coarse=((q[None]>>np.arange(3)[:,None,None])&1).astype(bool)
   bank1=NativeProbabilityBank(p1);bank4=NativeProbabilityBank(np.concatenate([coarse.astype(np.float64),p4[None]],axis=0))
   assert bank1.logical_pbits==6144 and bank4.logical_pbits==24576 and len(bank4.free_indices)==6144 and len(bank4.clamped_indices)==18432
   assert np.allclose(bank4.realized_probabilities[3],p4,rtol=0,atol=1e-15)
   (root/f'{layout}_program.npz').write_bytes(b'')
   with(root/f'{layout}_program.npz').open('wb')as f:np.savez_compressed(f,coarse_word=q,fraction_codes=codes)
   for kind,bank,Ts in [('K1',bank1,[32768]),('K4_fractional',bank4,[8192,32768])]:
    for T in Ts:
     mean=lo+span*(p1 if kind=='K1'else(q+p4)/8)
     variance=span**2*(p1*(1-p1)if kind=='K1'else p4*(1-p4)/64)/T
     bias=X@(mean-W).T;predicted=float(np.mean(bias*bias)+np.mean(X**2@variance.T))
     for seed in protocol['seeds']:
      states=bank.sample(seed,T,1)[0];counts=states.sum(axis=0,dtype=np.int32)
      if kind=='K1':u=counts.astype(np.float64)/T
      else:
       np.testing.assert_array_equal(counts[:3],coarse*T)
       u=(np.sum(counts[:3].astype(np.int64)*(2**np.arange(3))[:,None,None],axis=0)+counts[3])/(8*T)
      estimate=lo+span*u;MSE=float(np.mean((X@(estimate-W).T)**2));trajectory=hashlib.sha256(np.ascontiguousarray(states).tobytes()).hexdigest();del states
      file=root/f'{layout}_{kind}_T{T}_seed{seed}.npz'
      with file.open('xb')as f:np.savez_compressed(f,counts=counts,estimate=estimate)
      record={'layout':layout,'kind':kind,'T':T,'seed':seed,'logical_pbits':bank.logical_pbits,'free_pbits':len(bank.free_indices),'clamped_pbits':len(bank.clamped_indices),
       'all_node_observations':bank.logical_pbits*T,'free_node_updates':len(bank.free_indices)*T,'output_MSE':MSE,'predicted_output_MSE':predicted,'output_squared_bias':float(np.mean(bias*bias)),
       'file':str(file),'file_sha256':sha(file),'trajectory_sha256':trajectory}
      records.append(record);(root/'progress.json').write_text(json.dumps({'layout':layout,'kind':kind,'T':T,'seed':seed,'elapsed_seconds':time.monotonic()-begin})+'\n');print(json.dumps({'layout':layout,'kind':kind,'T':T,'seed':seed,'MSE':MSE}),flush=True)
  summary=[]
  for layout in ['storage_flat','transposed_matrix']:
   for kind,T in [('K1',32768),('K4_fractional',8192),('K4_fractional',32768)]:
    rows=[x for x in records if x['layout']==layout and x['kind']==kind and x['T']==T];assert len(rows)==5
    summary.append({'layout':layout,'kind':kind,'T':T,'mean_MSE':float(np.mean([x['output_MSE']for x in rows])),'predicted_MSE':rows[0]['predicted_output_MSE']})
  (root/'results.json').write_text(json.dumps({'status':'all30_native_strong_spatial_control_cases_complete_pending_audit','protocol_sha256':sha(root/'protocol.json'),'records':records,'summary':summary,'elapsed_seconds':time.monotonic()-begin,'scope':protocol['scope']},indent=2)+'\n')
 except Exception as error:
  (root/'failure.json').write_text(json.dumps({'type':type(error).__name__,'message':str(error)},indent=2)+'\n');raise
if __name__=='__main__':main()
