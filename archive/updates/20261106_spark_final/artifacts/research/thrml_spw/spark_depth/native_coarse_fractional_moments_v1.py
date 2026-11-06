"""Exact output-MSE variance accounting for reused weights across input vectors."""
import itertools,json,math
import numpy as np
from .prepare_v1 import OUT,sha


def mse_moments(X,bias,variance,k3,k4):
 gram=X.T@X;diagonal=np.diag(gram);denom=X.shape[0]*bias.shape[0];mean=var=0.
 for b,v,c3,c4 in zip(bias,variance,k3,k4):
  gb=gram@b;mean+=float(b@gb+diagonal@v)
  var+=float(2*v@(gram*gram)@v+(diagonal**2)@c4+4*(gb*gb)@v+4*(diagonal*gb)@c3)
 return mean/denom,var/(denom**2)


def fixture():
 X=np.array([[1.,.3],[-.5,2.]]);b=np.array([[.2,-.1]]);p=np.array([.2,.8]);scale=np.array([.7,1.2]);T=3;q=p*(1-p)
 mean,var=mse_moments(X,b,(scale**2*q/T)[None],(scale**3*q*(1-2*p)/T**2)[None],(scale**4*q*(1-6*q)/T**3)[None])
 values=[];probs=[]
 for counts in itertools.product(range(T+1),repeat=2):
  error=b[0]+scale*(np.array(counts)/T-p);values.append(float(np.mean((X@error)**2)))
  probs.append(math.prod(math.comb(T,k)*prob**k*(1-prob)**(T-k)for k,prob in zip(counts,p)))
 values=np.array(values);probs=np.array(probs);exact=float(probs@values);exact_var=float(probs@((values-exact)**2))
 np.testing.assert_allclose([mean,var],[exact,exact_var],rtol=1e-13)
 return {'exhaustive_outcomes':len(values),'mean':mean,'variance':var}


def main():
 root=OUT/'native_coarse_fractional_v1';r=json.loads((root/'results.json').read_text());audit=json.loads((OUT/'native_coarse_fractional_v1_audit.json').read_text());assert audit['result_sha256']==sha(root/'results.json');fixture_result=fixture()
 with np.load(OUT/'native_head_v1/inputs.npz')as f:d={k:f[k].copy()for k in f.files}
 X=d['hidden'].astype(np.float64);W=d['original_weights'].astype(np.float64);checks=[];weight_checks=[]
 for layout in ['storage_flat','transposed_matrix']:
  low=d[layout+'_lower'].astype(np.float64);span=d[layout+'_span'].astype(np.float64)
  with np.load(root/f'{layout}_program.npz')as f:q=f['coarse_word'].copy();codes=f['fraction_codes'].copy()
  fields=codes.astype(np.float32)*np.float32(8/4095)-np.float32(4);p4=1/(1+np.exp(-2*fields.astype(np.float64)))
  f1=d[layout+'_codes'].astype(np.float32)*np.float32(8/4095)-np.float32(4);p1=1/(1+np.exp(-2*f1.astype(np.float64)))
  for kind,T in [('K1',32768),('K4_fractional',8192),('K4_fractional',32768)]:
   prob=p1 if kind=='K1'else p4;scale=span if kind=='K1'else span/8;mean_weight=low+span*(p1 if kind=='K1'else(q+p4)/8);pq=prob*(1-prob)
   v=scale**2*pq/T;c3=scale**3*pq*(1-2*prob)/T**2;c4=scale**4*pq*(1-6*pq)/T**3
   mean,var=mse_moments(X,mean_weight-W,v,c3,c4);rows=[x for x in r['records']if x['layout']==layout and x['kind']==kind and x['T']==T]
   observed=float(np.mean([x['output_MSE']for x in rows]));z=(observed-mean)/np.sqrt(var/5);assert abs(z)<8
   for row in rows:
    np.testing.assert_allclose(mean,row['predicted_output_MSE'],rtol=1e-12)
    with np.load(row['file'])as f:e=f['estimate']-mean_weight
    expected=v.sum();variance=float(np.sum(2*v*v+c4));zw=(float(np.sum(e*e))-expected)/np.sqrt(variance);assert abs(zw)<8
    weight_checks.append({'layout':layout,'kind':kind,'T':T,'seed':row['seed'],'weight_SSE_z':float(zw)})
   checks.append({'layout':layout,'kind':kind,'T':T,'expected_output_MSE':mean,'variance_of_one_draw_output_MSE':var,'mean_measured_output_MSE':observed,'five_draw_mean_MSE_z':float(z)})
 output={'status':'all30_weight_moments_and_six_correlated_output_MSE_moments_passed','source_sha256':sha(__file__),'result_sha256':sha(root/'results.json'),'audit_sha256':sha(OUT/'native_coarse_fractional_v1_audit.json'),'exact_enumeration_fixture':fixture_result,'output_checks':checks,'weight_checks':weight_checks,
 'scope':'Exact finite-binomial moment calculation includes covariance between fixed input vectors sharing each sampled weight. Monte Carlo output-MSE means need not equal their expectations with only five draws. No neural or hardware systems claim.'}
 (OUT/'native_coarse_fractional_moments_v1.json').write_text(json.dumps(output,indent=2)+'\n');print(json.dumps(checks,indent=2))
if __name__=='__main__':main()
