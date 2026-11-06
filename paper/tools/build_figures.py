"""Rebuild publication plots entirely from the bundled audited JSON inputs."""
from pathlib import Path
import json, hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
def read(s):return json.loads((ROOT/'evidence'/s).read_text())
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,'savefig.bbox':'tight'})
blue='#245a81';orange='#b66523';green='#297366'
def save(fig,name):
 fig.savefig(ROOT/'figures'/f'{name}.pdf');fig.savefig(ROOT/'figures'/f'{name}.png',dpi=220);plt.close(fig)
# Complete task quality and matched-layout control.
d=read('cross_model_summary_v2/results.json');g=read('benchmark_summary_v1/results.json')
labels=['GPT-2: original grouping','GPT-2: selected grouping','DistilBERT: selected grouping','LFM3: selected grouping']
points=[g['methods']['matched_flat']]+d['models']
ratios=[points[0]['perplexity_ratio']]+[p['perplexity_ratio']for p in points[1:]]
ci=[points[0]['MC95_PPL_ratio_interval']]+[p['MC95_perplexity_ratio_interval']for p in points[1:]]
fig,ax=plt.subplots(figsize=(7.5,3.05));y=np.arange(4)
for i,(v,c)in enumerate(zip(ratios,ci)):
 ax.errorbar(v,i,xerr=[[v-c[0]],[c[1]-v]],fmt='o',color=orange if i==0 else blue,capsize=4)
ax.set_yticks(y,labels);ax.invert_yaxis();ax.axvline(1,color='0.55',lw=1);ax.axvline(1.02,color=orange,ls='--',label='2% quality margin');ax.set_xlabel('Perplexity ratio to the original BF16 model (five-seed 95% t interval)');ax.legend(loc='lower right',frameon=False);ax.grid(axis='x',alpha=.15);save(fig,'quality')
# Training causal diagnostic: no confidence interval implied.
d=read('tied_head_v1/results.json')['summaries'];keys=['BF16','flat','restore_input_use_only','restore_output_use_only','restore_both_uses','only_embedding_reoriented','all_reoriented']
labels=['BF16','Original grouping','Restore input use','Restore output use','Restore both uses','Reorient tied matrix','Reorient all matrices']
fig,ax=plt.subplots(figsize=(7.5,3.5));vals=[d[k]['PPL']for k in keys];ax.barh(labels[::-1],vals[::-1],color=[blue]*5+[orange,green]);ax.axvline(vals[0],ls='--',color='0.3',lw=1);ax.set_xlabel('Training diagnostic perplexity (one fixed weight draw)');ax.set_xlim(0,68)
for i,v in enumerate(vals[::-1]):ax.text(v+.5,i,f'{v:.2f}',va='center',fontsize=9)
save(fig,'head_intervention')
# Actual reconstruction budgets and quality.
d=read('reduced_work_summary_v1/results.json');fig,ax=plt.subplots(figsize=(6.6,3.3))
for k,x,color,label in [('uniform',1,blue,'Uniform T = 32,768'),('reduced',d['observation_fraction'],green,'Sensitivity allocation')]:
 q=d[k];ci=q.get('MC95_t_PPL_ratio_interval',q.get('MC95_t4_PPL_ratio_interval'));v=q['PPL_ratio'];ax.errorbar(x,v,yerr=[[v-ci[0]],[ci[1]-v]],fmt='o',color=color,capsize=5,label=label)
ax.axhline(1,color='0.6',lw=1);ax.axhline(1.02,color=orange,ls='--',label='2% quality margin');ax.set(xlim=(.48,1.12),ylim=(.992,1.029),xlabel='Nominal reconstruction observations / uniform budget',ylabel='GPT-2 perplexity / BF16 perplexity');ax.legend(loc='upper left',frameon=False,fontsize=9);save(fig,'allocation')
# Strong native control, no CI fabrication.
s=read('native_coarse_fractional_v1_audit.json')['summary'];fig,axs=plt.subplots(1,2,figsize=(7.5,3.2),sharey=True)
for ax,layout,title in zip(axs,['storage_flat','transposed_matrix'],['Original grouping','Reoriented grouping']):
 rows=[v for v in s if v['layout']==layout];xx=np.arange(len(rows));ax.bar(xx,[v['mean_MSE']for v in rows],color=[blue,orange,green]);ax.scatter(xx,[v['predicted_MSE']for v in rows],marker='_',s=170,color='black',label='Analytical expectation',zorder=3);ax.set_yscale('log');ax.set_xticks(xx,['K1\nT=32,768','K4\nT=8,192','K4\nT=32,768']);ax.set_title(title);ax.grid(axis='y',alpha=.15)
axs[0].set_ylabel('Output MSE (five-draw mean)');axs[1].legend(loc='upper right',fontsize=8,frameon=False);save(fig,'native_comparison')
# Gaussian and device controls are independent cohorts.
noise=read('gaussian_control_v3_audit.json')['statistics'];fig,ax=plt.subplots(figsize=(6.4,2.3))
for i,(k,v)in enumerate(noise.items()):
 r=v['PPL_ratio_to_baseline'];lo,hi=v['MC95_t4_interval'];ax.errorbar(r,i,xerr=[[r-lo],[hi-r]],fmt='o',capsize=5,color=blue if i==0 else orange)
ax.set_yticks([0,1],['Bernoulli reconstruction','Matched Gaussian noise']);ax.axvline(1,color='0.6',lw=1);ax.set_xlabel('Training perplexity / BF16 perplexity (five-seed 95% t interval)');ax.set_ylim(-.6,1.6);save(fig,'gaussian_control')
print('Rebuilt five publication figures from bundled evidence.')
