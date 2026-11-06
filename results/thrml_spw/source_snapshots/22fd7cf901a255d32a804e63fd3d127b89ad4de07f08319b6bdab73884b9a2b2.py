"""Reproduce Torx's published PSWAP quickstart without changing pinned venvs."""
from pathlib import Path
import hashlib,json,datetime
import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import binomtest
from torx import psc
OUT=Path('results/thrml_spw/torx_probe/official_quickstart')
def main():
    OUT.mkdir(exist_ok=True);assert not (OUT/'result.json').exists();theta=jnp.asarray([jnp.log(.3/.7)]);circuit=psc.DiscretePCircuit([psc.PSWAP([0,1])]);bs=psc.BranchingSimulator(num_samples=20000);compiled=bs.build_circuit(circuit,[theta]);assert compiled.num_pdits==2;sample=np.asarray(bs.sample(compiled,jnp.array([1,0],jnp.int32),jax.random.key(0)));stay=np.all(sample==np.array([1,0]),axis=1);swap=np.all(sample==np.array([0,1]),axis=1);assert np.all(stay|swap)
    sv=psc.StateVectorSimulator();density=np.asarray(sv.density(sv.build_circuit(circuit,[theta]),jnp.asarray([0.,0.,1.,0.])));q=float(jax.nn.sigmoid(theta[0]));np.testing.assert_allclose(density,[0,q,1-q,0],rtol=1e-6,atol=1e-7);test=float(binomtest(int(swap.sum()),20000,q).pvalue);assert test>.001
    np.savez_compressed(OUT/'samples.npz',samples=sample,density=density,theta=np.asarray(theta));r={'recorded_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'passed':True,'official_source':'https://docs.torx.ai/','logical_sites':2,'logical_gate_applications':20000,'seed':0,'nominal_swap_probability':q,'observed_swap_fraction':float(swap.mean()),'observed_stay_fraction':float(stay.mean()),'two_sided_binomial_pvalue':test,'frozen_binomial_threshold':.001,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'scope':'Published conditional-sampling quickstart, one circuit application per sample. Not a one-site weight representation or stationary-occupancy experiment.'};(OUT/'result.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
if __name__=='__main__':main()
