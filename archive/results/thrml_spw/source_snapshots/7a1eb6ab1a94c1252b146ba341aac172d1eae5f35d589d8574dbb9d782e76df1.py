"""Exact finite-channel information and Bayes risk, not a native experiment."""
from pathlib import Path
import json,hashlib
from importlib.metadata import version
import numpy as np
from scipy.stats import binom
OUT=Path('results/thrml_spw/information_budget')
BUDGETS=[1,2,4,8,16,32,64,128,256,512,1024,4096]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def measures(p,likelihood,target=None):
    target=p if target is None else target;mixture=likelihood.mean(0);positive=mixture>0;entropy=-float(np.sum(mixture[positive]*np.log2(mixture[positive])));mask=likelihood>0;terms=np.zeros_like(likelihood);terms[mask]=likelihood[mask]*np.log2(likelihood[mask]);conditional=-float(terms.sum()/len(p));posterior=(target[:,None]*likelihood).mean(0)/mixture;second=(target[:,None]**2*likelihood).mean(0)/mixture;risk=float(np.sum(mixture*np.maximum(second-posterior**2,0)));return {'mutual_information_bits':entropy-conditional,'Bayes_MMSE':risk}

def main():
    assert not (OUT/'manifest.json').exists();OUT.mkdir(parents=True,exist_ok=True);sources={str(Path('research/thrml_spw')/n):sha(Path('research/thrml_spw')/n) for n in ['information_budget.py','verify_information_budget.py','information_budget_report.py']};protocol={'field_bounds':[4,2],'one_node_control_codes':256,'control_prior':'uniform over256 field codes, not uniform weights or observed neural controls','positional_K':[2,4,8],'positional_prior':'uniform over all2^K positional words; independent equiprobable desired bit labels','budgets':BUDGETS,'max_updates_for_spatial_comparisons':4096,'quantized_fields':'F32 uniformly spaced [-B,B], probabilities evaluated in F64 by official unary sigmoid(2b) mapping','scope':'Exact channel calculation from verified stationary probabilities. Counts sufficient for IID Bernoulli observations; no new native samples, neural scores, decoder deployment, controller-free storage or physical cost proof. Bayes MMSE assumes known label prior and optimal decoder.','source_sha256':sources};(OUT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');rows=[]
    for b in protocol['field_bounds']:
        fields=(-b+np.arange(256,dtype=float)*2*b/255).astype(np.float32);p=np.exp(-np.logaddexp(0,-2*fields.astype(float)));epsilon=float(1/(1+np.exp(2*b)));bitp=np.array([epsilon,1-epsilon])
        for t in BUDGETS:
            count=np.arange(t+1);full=binom.pmf(count[None,:],t,p[:,None]);binary=binom.pmf(count[None,:],t,bitp[:,None]);np.testing.assert_allclose(full.sum(1),1,rtol=2e-12,atol=2e-12);np.testing.assert_allclose(binary.sum(1),1,rtol=2e-12,atol=2e-12);name=OUT/f'B{b}_T{t}.npz';np.savez_compressed(name,fields=fields,probabilities=p,binomial_likelihood=full,binary_probabilities=bitp,binary_likelihood=binary);r=measures(p,full);rows.append({'field_bound':b,'K':1,'samples':t,'node_updates':t,'control_entropy_bits':8.,'count_alphabet_log2':float(np.log2(t+1)),'unbiased_frequency_MSE':float(np.mean(p*(1-p))/t),**r})
            info=measures(bitp,binary,target=np.array([0.,1.]))
            for k in protocol['positional_K']:
                if k*t>4096:continue
                coeff=2.**np.arange(k)/((1<<k)-1);rows.append({'field_bound':b,'K':k,'samples':t,'node_updates':k*t,'control_entropy_bits':float(k),'count_alphabet_log2':float(k*np.log2(t+1)),'mutual_information_bits':k*info['mutual_information_bits'],'Bayes_MMSE':float(np.sum(coeff**2)*info['Bayes_MMSE'])})
    (OUT/'statistics.json').write_text(json.dumps({'rows':rows,'likelihood_hashes':{str(f):sha(f) for f in sorted(OUT.glob('B*_T*.npz'))}},indent=2)+'\n');manifest={'protocol_sha256':sha(OUT/'protocol.json'),'statistics_sha256':sha(OUT/'statistics.json'),'new_native_samples':0,'new_neural_scores':0,'row_count':len(rows),'packages':{n:version(n) for n in ['numpy','scipy']},'source_sha256':sources,'scope':protocol['scope']};(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
