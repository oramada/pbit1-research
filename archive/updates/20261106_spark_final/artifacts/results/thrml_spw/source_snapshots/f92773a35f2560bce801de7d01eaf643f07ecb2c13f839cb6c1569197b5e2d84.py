"""Native all-free finite-field positional and one-p-bit scalar experiment."""
import json,time,os
import numpy as np
import jax
from .native import NativeProbabilityBank
from .finite_positional_common import OUT,sha,parameters,seed_for,source_guard,metrics
from .provenance import manifest
def main():
    p=json.loads((OUT/'protocol.json').read_text());source_guard(p);assert not (OUT/'manifest.json').exists();assert jax.default_backend()=='gpu';rows=[];updates=0;start_time=time.monotonic()
    for bound in p['field_bounds']:
        for k in [1]+p['spatial_bits']:
            name=f'B{bound}_K{k}';assert not (OUT/f'{name}.npz').exists();target,fields,codes,coeff,prob=parameters(p,bound,k);bank=NativeProbabilityBank(prob);np.testing.assert_array_equal(bank.biases.reshape(fields.shape),fields);assert bank.clamped_indices.size==0
            budgets=p['single_samples'] if k==1 else p['parallel_samples'];counts=np.empty((len(p['seeds']),p['trials_per_seed'],len(budgets),len(target),k),np.uint16)
            for si,seed in enumerate(p['seeds']):
                for first in range(0,p['trials_per_seed'],p['chain_batch']):
                    n=min(p['chain_batch'],p['trials_per_seed']-first);bits=bank.sample(seed_for(p,bound,k,seed,first),max(budgets),n);prefix=np.cumsum(bits,axis=1,dtype=np.uint16);counts[si,first:first+n]=prefix[:,np.array(budgets)-1];updates+=n*max(budgets)*len(target)*k
                print(f'{name} seed={seed} complete elapsed={time.monotonic()-start_time:.1f}s',flush=True)
            path=OUT/f'{name}.npz';np.savez_compressed(path,targets=target,fields=fields,codes=codes,probabilities=prob,coefficients=coeff,budgets=budgets,counts=counts)
            for i,t in enumerate(budgets):rows.append({'condition':name,'field_bound':bound,**metrics(counts[:,:,i],target,prob,coeff,t)})
            source_guard(p)
    result={'rows':rows,'native_node_updates':updates,'all_free_nodes':True,'passed':all(abs(r['mse_z'])<p['gate_max_abs_mse_z'] for r in rows),'max_abs_mse_z':max(abs(r['mse_z']) for r in rows),'protocol_sha256':sha(OUT/'protocol.json'),'raw_sha256':{str(f):sha(f) for f in sorted(OUT.glob('B*_K*.npz'))},'XLA_FLAGS':os.environ.get('XLA_FLAGS'),'backend':jax.default_backend(),'elapsed_seconds':time.monotonic()-start_time};(OUT/'statistics.json').write_text(json.dumps(result,indent=2)+'\n');manifest(OUT/'manifest.json',p,result);print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2));assert result['passed'],'Original six-sigma scalar gate failed; preserve evidence.'
if __name__=='__main__':main()
