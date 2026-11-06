"""Independent mode-normalized binomial recurrences, KL information and Bayes risk."""
from pathlib import Path
import math,json
import numpy as np
from .information_budget import OUT,sha

def likelihood(p,t):
    result=np.zeros((len(p),t+1))
    for row,q in enumerate(p):
        mode=min(t,math.floor((t+1)*q));result[row,mode]=1
        for c in range(mode-1,-1,-1):result[row,c]=result[row,c+1]*(c+1)/(t-c)*(1-q)/q
        for c in range(mode+1,t+1):result[row,c]=result[row,c-1]*(t-c+1)/c*q/(1-q)
        result[row]/=result[row].sum()
    return result

def values(probability,pmf,target):
    mixture=pmf.mean(0);mi=0.;risk=0.
    for count,mass in enumerate(mixture):
        if mass==0:continue
        column=pmf[:,count];positive=column>0;mi+=float(np.sum(column[positive]*np.log2(column[positive]/mass))/len(probability));posterior=column/column.sum();mean=float(posterior@target);risk+=float(mass*(posterior@((target-mean)**2)))
    return mi,risk

def main():
    p=json.loads((OUT/'protocol.json').read_text());m=json.loads((OUT/'manifest.json').read_text());r=json.loads((OUT/'statistics.json').read_text());assert m['protocol_sha256']==sha(OUT/'protocol.json') and m['statistics_sha256']==sha(OUT/'statistics.json');checked=0;elements=0
    for source,h in p['source_sha256'].items():assert sha(source)==h
    for path,h in r['likelihood_hashes'].items():
        assert sha(path)==h;a=np.load(path);b=int(Path(path).stem.split('_')[0][1:]);t=int(Path(path).stem.split('_')[1][1:]);fields=np.array([-b+c*2*b/255 for c in range(256)],np.float32);np.testing.assert_array_equal(fields,a['fields']);q=np.array([1/(1+math.exp(-2*float(v))) for v in fields]);np.testing.assert_allclose(q,a['probabilities'],rtol=1e-14,atol=1e-16);full=likelihood(q,t);eps=1/(1+math.exp(2*b));binary=likelihood(np.array([eps,1-eps]),t);np.testing.assert_allclose(full,a['binomial_likelihood'],rtol=3e-10,atol=2e-14);np.testing.assert_allclose(binary,a['binary_likelihood'],rtol=3e-10,atol=2e-14);elements+=full.size+binary.size;mi,risk=values(q,full,q);row=next(z for z in r['rows'] if z['field_bound']==b and z['model']=='single256' and z['samples']==t);np.testing.assert_allclose(row['mutual_information_bits'],mi,rtol=1e-10,atol=2e-12);np.testing.assert_allclose(row['Bayes_MMSE'],risk,rtol=1e-7,atol=2e-15);assert risk<=row['unbiased_frequency_MSE']+1e-12 and -1e-12<=mi<=min(8,math.log2(t+1))+1e-10;checked+=1
        target16=np.arange(16)/15;np.testing.assert_allclose(target16,a['desired_weights16'],rtol=0,atol=2e-16);clipped=np.clip(target16,eps,1-eps);fc=np.array([.5*math.log(float(z)/(1-float(z))) for z in clipped]);codes16=np.rint((np.clip(fc,-b,b)+b)*255/(2*b)).astype(np.uint8);np.testing.assert_array_equal(codes16,a['codes16']);p16=q[codes16];np.testing.assert_allclose(p16,a['probabilities16'],rtol=1e-14,atol=1e-16);pmf16=likelihood(p16,t);np.testing.assert_allclose(pmf16,a['single16_likelihood'],rtol=3e-10,atol=2e-14);elements+=pmf16.size;mi16,risk16=values(p16,pmf16,target16);row16=next(z for z in r['rows'] if z['field_bound']==b and z['model']=='single16' and z['samples']==t);np.testing.assert_allclose(row16['mutual_information_bits'],mi16,rtol=1e-10,atol=2e-12);np.testing.assert_allclose(row16['Bayes_MMSE'],risk16,rtol=1e-7,atol=2e-15);assert -1e-12<=mi16<=min(4,math.log2(t+1))+1e-10;checked+=1;bitmi,bitrisk=values(np.array([eps,1-eps]),binary,np.array([0.,1.]))
        if t==1:
            entropy=-eps*math.log2(eps)-(1-eps)*math.log2(1-eps);np.testing.assert_allclose(bitmi,1-entropy,rtol=1e-12,atol=1e-12)
        for k in [2,4,8]:
            if k*t>4096:continue
            row=next(z for z in r['rows'] if z['field_bound']==b and z['K']==k and z['samples']==t);np.testing.assert_allclose(row['mutual_information_bits'],k*bitmi,rtol=1e-10,atol=2e-12);coefficient_square_sum=((4**k-1)/3)/((2**k-1)**2);np.testing.assert_allclose(row['Bayes_MMSE'],coefficient_square_sum*bitrisk,rtol=1e-6,atol=2e-15);assert -1e-12<=k*bitmi<=min(k,k*math.log2(t+1))+1e-10;checked+=1
    for b in [4,2]:
        for model in ['single256','single16','positional2','positional4','positional8']:
            rows=sorted([z for z in r['rows'] if z['field_bound']==b and z['model']==model],key=lambda z:z['samples']);assert np.all(np.diff([z['mutual_information_bits'] for z in rows])>=-1e-10);assert np.all(np.diff([z['Bayes_MMSE'] for z in rows])<=1e-12)
    assert checked==m['row_count'];result={'passed':True,'rows_independently_verified':checked,'binomial_likelihood_elements_checked':elements,'all_information_bounds_and_monotonicity_passed':True,'KL_and_posterior_squared_deviation_verified':True,'new_native_samples':0,'producer_manifest_sha256':sha(OUT/'manifest.json'),'source_sha256':sha(__file__),'scope':'Exact analytic finite-code-prior channel calculation; no new neural or hardware evidence.'};(OUT/'evidence_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
