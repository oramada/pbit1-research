"""Plot independently audited matched-resource cohorts; no raw model assets."""
import argparse
from pathlib import Path
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    root=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',type=Path,default=root/'benchmarks/matched_resources_20261105.json')
    parser.add_argument('--audit',type=Path,default=root/'benchmarks/matched_resources_20261105.audit.json')
    parser.add_argument('--output',type=Path,default=root/'docs/assets')
    args=parser.parse_args();r=json.loads(args.input.read_text());a=json.loads(args.audit.read_text())
    assert hashlib.sha256(args.input.read_bytes()).hexdigest()==a['statistics_sha256']
    assert a['cases_verified']==45
    args.output.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'svg.fonttype':'none',
        'figure.facecolor':'#f8fafc','axes.facecolor':'#f8fafc','text.color':'#12213a',
        'axes.spines.top':False,'axes.spines.right':False})
    fig,(ax,bx)=plt.subplots(1,2,figsize=(13.4,4.7))
    colors={'K1':'#089b88','K4':'#cf782d','K8':'#5260ce'}
    for key in ['K1','K4','K8']:
        means=np.array([c['methods'][key]['PPL_ratio_to_own_same_fit_BF16'] for c in r['cohorts']])
        ci=np.array([c['methods'][key]['PPL_ratio_MC95'] for c in r['cohorts']])
        ax.errorbar(np.arange(3)+{'K1':-.045,'K4':0,'K8':.045}[key],means,
            yerr=[means-ci[:,0],ci[:,1]-means],fmt='o-',capsize=4,color=colors[key],label=key+' ('+str(int(key[1:]))+' p-bit/parameter)')
    ax.axhline(1.02,color='#64748b',ls='--',lw=1,label='2% upper-bound criterion')
    ax.axhline(1,color='#cbd5e1',lw=.8)
    ax.set(xticks=range(3),xticklabels=['C = 1,024','C = 4,096','C = 8,192'],ylim=(.95,1.25),
        ylabel='PPL ratio to own-backend same-fit BF16',xlabel='Matched retained-work budget: C = K × T')
    ax.set_title('Quality at equal scheduled work',loc='left',weight='bold',pad=15)
    ax.legend(frameon=False,fontsize=9,loc='center right');ax.grid(axis='y',alpha=.15)
    minima=r['lowest_qualifying_budget_among_explored'];k1=minima['K1'];k8=minima['K8']
    values=[1/8,k1['retained_work_budget']/k8['retained_work_budget'],k1['samples_per_pbit']/k8['samples_per_pbit']]
    positions=np.arange(3)
    bx.bar(positions-.17,values,width=.32,color=colors['K1'],label='K1, C8192')
    bx.bar(positions+.17,[1,1,1],width=.32,color=colors['K8'],label='K8, C1024')
    bx.set_yscale('log',base=2)
    bx.set(ylim=(1/16,128),xticks=positions,xticklabels=['Logical\np-bits','Scheduled\nwork','Temporal\nsamples'],ylabel='Relative to lowest qualifying K8 condition')
    for x,y,label in zip(positions-.17,values,['1/8×','8×','64×']):bx.text(x,y*1.3,label,ha='center',color=colors['K1'],weight='bold')
    bx.set_title('Lowest qualifying explored budgets',loc='left',weight='bold',pad=15)
    bx.legend(frameon=False,loc='upper left',fontsize=9);bx.grid(axis='y',alpha=.12)
    fig.subplots_adjust(bottom=.28,top=.88,wspace=.30)
    fig.text(.07,.06,'45 accepted cases · five seeds per format/budget · fixed 2,048-target LFM3 subset. Bars are conditional MC95 t4 intervals.\nC1024 uses GB10; higher-budget K1 uses GB10 and K4/K8 use H100 with own-backend controls. Backend provenance is retained.\nLogical counts assume full residency; scheduled work and temporal samples are not measured hardware energy or latency.',fontsize=9,color='#64748b')
    for ext in ['svg','png']:fig.savefig(args.output/f'matched_resources_20261105.{ext}',dpi=180,bbox_inches='tight')


if __name__=='__main__':main()
