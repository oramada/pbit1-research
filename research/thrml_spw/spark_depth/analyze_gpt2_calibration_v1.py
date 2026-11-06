"""Explain irreducible calibration error and the cost of the oracle control."""
import json
import numpy as np
from .prepare_v1 import OUT,sha

def main():
 root=OUT/'gpt2_calibration_control_v1';r=json.loads((root/'results.json').read_text());audit=json.loads((OUT/'gpt2_calibration_control_v1_audit.json').read_text());assert audit['result_sha256']==sha(root/'results.json')
 kinds=['ideal','uncalibrated','oracle'];groups={k:[x for x in r['cases']if x['kind']==k]for k in kinds};pairs={};N=124439808
 for k in kinds[1:]:
  delta=np.array([x['NLL']for x in groups[k]])-np.array([x['NLL']for x in groups['ideal']]);half=2.7764451051977987*delta.std(ddof=1)/np.sqrt(5)
  pairs[k]={'PPL_ratio_to_ideal':float(np.exp(delta.mean())),'paired_MC95_t4':np.exp([delta.mean()-half,delta.mean()+half]).tolist()}
 bias=np.array([np.mean([x['bias_SSE_relative_to_requested_mean']for x in groups[k]])for k in kinds]);variance=np.array([np.mean([x['sampling_variance_SSE']for x in groups[k]])for k in kinds])
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained');labels=['Ideal device','Uncalibrated','Oracle correction']
 for i,k in enumerate(kinds):
  s=audit['statistics'][k];v=s['PPL_ratio'];lo,hi=s['MC95_t4'];axes[0].errorbar(v,i,xerr=[[v-lo],[hi-v]],fmt='o',capsize=4,color=['#4477aa','#cc6677','#228833'][i])
 axes[0].axvline(1,color='gray',ls=':');axes[0].set(yticks=range(3),yticklabels=labels,xlabel='Perplexity ratio to CPU reference',title='Five draws; 1,024 training targets')
 x=np.arange(3);axes[1].bar(x,variance/variance[0],label='Sampling variance');axes[1].bar(x,bias/variance[0],bottom=variance/variance[0],label='Persistent calibration bias')
 axes[1].set(xticks=x,xticklabels=['Ideal','Uncalibrated','Oracle'],ylabel='Expected weight SSE / ideal SSE',title='More samples do not remove bias');axes[1].legend()
 dest=OUT/'gpt2_calibration_analysis_v1';dest.mkdir(exist_ok=True);fig.savefig(dest/'device_error.png',dpi=180);fig.savefig(dest/'device_error.svg');plt.close(fig)
 out={'status':'audited_training_device_error_summary','source_sha256':sha(__file__),'audit_sha256':sha(OUT/'gpt2_calibration_control_v1_audit.json'),'paired_comparisons':pairs,
 'mean_bias_SSE':dict(zip(kinds,bias.tolist())),'mean_sampling_variance_SSE':dict(zip(kinds,variance.tolist())),'bias_reduction_factor_by_oracle':float(bias[1]/bias[2]),
 'limiting_weight_RMSE_as_T_increases_uncalibrated':float(np.sqrt(bias[1]/N)),'calibration_table_bytes_if_two_FP32_values_retained_per_device':8*N,
 'mean_clipped_oracle_requests':float(np.mean([x['clipped_inverse_requests']for x in groups['oracle']])),
 'scope':'Illustrative gainSD.02 and logit-offsetSD.02. All learned scalars remain K1, but oracle calibration knows true device coefficients; storing those costs extra bytes. A small CPU training diagnostic, not measured hardware, no statistical-equivalence proof.'}
 (dest/'results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
if __name__=='__main__':main()
