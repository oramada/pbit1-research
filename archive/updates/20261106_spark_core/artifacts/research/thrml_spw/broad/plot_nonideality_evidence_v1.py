"""Standalone figures for two separate bounded synthetic-device studies."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def main():
    root=Path('results/thrml_spw/broad_verification_20261105/nonideality_evidence_v1');r=json.loads((root/'results.json').read_text())
    plt.rcParams.update({'font.size':10,'svg.fonttype':'none'})
    fig,(a,b)=plt.subplots(1,2,figsize=(11,4.7),layout='constrained')
    rows=r['calibration']['conditions'];x=list(range(3))
    ratios=[z['perplexity_ratio_to_FP32']for z in rows];upper=[z['original_crossed_bootstrap_ratio_upper']for z in rows]
    a.bar(x,ratios,color=['#1b9e77','#d95f02','#7570b3'],width=.62)
    a.scatter(x,upper,color='black',marker='_',s=220,zorder=4,label='Crossed bootstrap upper')
    a.axhline(1.05,color='#555555',linestyle='--',linewidth=1,label='Original 5% quality margin')
    a.set_yscale('log');a.set_ylim(.8,16);a.set_xticks(x,['Ideal','Uncalibrated','4,096-probe\ncalibration'])
    a.set_ylabel('Perplexity / original FP32 reference')
    a.set_title('Selected DistilGPT2 subset',loc='left',weight='bold')
    for i,v in enumerate(ratios):a.text(i,v*1.09,f'{v:.3f}×',ha='center',fontsize=9)
    a.legend(loc='upper left',frameon=False,fontsize=8);a.grid(axis='y',alpha=.15)
    colors={0.:'#1b9e77',.01:'#7570b3',.04:'#d95f02'}
    for sigma,color in colors.items():
        for policy,style in [('retain','-'),('refresh','--')]:
            data=[z for z in r['drift']['groups']if z['drift_SD_per_512sweep_interval']==sigma and z['policy']==policy]
            if policy=='refresh':data=[next(z for z in r['drift']['groups']if z['drift_SD_per_512sweep_interval']==sigma and z['age_intervals']==0),*data]
            data.sort(key=lambda z:z['age_intervals'])
            xx=[z['age_intervals']for z in data];yy=[z['mean_MSE']for z in data];err=[z['approx_normal95_halfwidth']for z in data]
            b.errorbar(xx,yy,yerr=err,color=color,linestyle=style,marker='o',markersize=4,capsize=2)
    b.set_yscale('log');b.set_xticks([0,8,32]);b.set_xlabel('Age after initial calibration (512-sweep intervals)')
    b.set_ylabel('Operator output MSE');b.set_title('Separate drift-retention operator',loc='left',weight='bold');b.grid(alpha=.15)
    handles=[Line2D([0],[0],color=c,label=f'Drift SD {s:g}')for s,c in colors.items()]
    handles += [Line2D([0],[0],color='#555555',linestyle='-',label='Retain controls'),Line2D([0],[0],color='#555555',linestyle='--',label='Refresh controls')]
    b.legend(handles=handles,frameon=False,fontsize=8,loc='upper left')
    fig.suptitle('Synthetic device errors remain a practical boundary',fontsize=14,weight='bold')
    fig.text(.01,-.06,'Left: 25.922% of parameters converted; five joint device/sampling seeds, T=1,024. Calibration still fails the original gate.\nRight: 2.36M-weight operator, ten joint seeds; approximate normal 95% bars. Refresh adds 4,096 probe sweeps. No physical hardware claim.',fontsize=9,color='#475467')
    for ext in ['svg','png']:fig.savefig(root/f'nonideality_limits.{ext}',bbox_inches='tight',dpi=160,metadata={'Date':None}if ext=='svg'else None)
    plt.close(fig);print(root/'nonideality_limits.svg')

if __name__=='__main__':main()
