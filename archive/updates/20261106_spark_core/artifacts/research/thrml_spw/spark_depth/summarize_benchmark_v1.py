"""Render only the completed, independently audited benchmark cohort."""
import json,math
import numpy as np
from .prepare_v1 import OUT,sha

def main():
    audit_path=OUT/'benchmark_audit_v1/results.json';a=json.loads(audit_path.read_text());assert a['status']=='all12cases_saved_array_coverage_and_statistics_audited'
    assert a['benchmark_results_sha256']==sha(OUT/'benchmark_v1/results.json')
    dest=OUT/'benchmark_summary_v1';dest.mkdir(exist_ok=False);cases=a['cases'];baseline=cases[0];summary={}
    for kind in ['selected','matched_flat']:
        rows=[r for r in cases if r['case']['kind']==kind];delta=np.array([r['NLL']-baseline['NLL']for r in rows]);stats=a['statistics'][kind]
        summary[kind]={'perplexity_at_mean_NLL':math.exp(float(np.mean([r['NLL']for r in rows]))),'perplexity_ratio':stats['PPL_ratio'],
            'MC95_PPL_ratio_interval':stats['MC95_t_PPL_ratio_interval'],'joint_article_seed_PPL_ratio_interval':stats['secondary_article_and_seed_bootstrap_PPL_ratio_interval'],
            'passes_frozen_2pct_gate':stats['primary_2pct_pass'],'seed_count':len(rows)}
    peak=max(c['maximum_GPU_allocation_bytes']for c in cases);assert peak<=2*1024**3
    report={'status':'completed_full_benchmark_summary','source_sha256':sha(__file__),'audit_sha256':sha(audit_path),'BF16_perplexity':baseline['PPL'],
        'finite_control_mean_perplexity':cases[1]['PPL'],'methods':summary,'reviews_or_articles':62,'targets_per_case':283225,'cases':12,'parameter_count':124439808,
        'samples_per_parameter':32768,'logical_pbits_per_parameter':1,'max_Torch_GPU_allocation_bytes':peak,
        'scope':'OnecompleteidealBinomialR1weightdraw reusedacrossfullbenchmark; previouslyinspectedWikiText2testre-evaluation. Complete5seedcohorts, peroperatorlayouttrained/calibratedbeforetest. NotphysicalhardwareorfullmodelnativeTHRML; no universalBF16equalityclaim.'}
    (dest/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,5));names=['Original grouping','Operator-aware grouping']
    for i,kind in enumerate(['matched_flat','selected']):
        d=summary[kind];lo,hi=d['MC95_PPL_ratio_interval'];m=d['perplexity_ratio'];axes[0].errorbar(i,m,yerr=[[m-lo],[hi-m]],fmt='o',capsize=7,color=['tomato','teal'][i])
    axes[0].set_xticks([0,1],names);axes[0].axhline(1,color='gray',linestyle=':',label='BF16');axes[0].axhline(1.02,color='orange',linestyle='--',label='Frozen 2% gate');axes[0].set(ylabel='Perplexity / BF16',title='Complete five-seed cohorts; 95% MC intervals');axes[0].legend()
    n=np.array(baseline['article_target_counts']);base=np.array(baseline['article_NLL_sums'])/n
    for kind,color,label in [('matched_flat','tomato',names[0]),('selected','teal',names[1])]:
        rows=[r for r in cases if r['case']['kind']==kind];mean=np.mean([r['article_NLL_sums']for r in rows],axis=0)/n
        axes[1].scatter(base,mean-base,s=14+50*n/n.max(),alpha=.6,color=color,label=label)
    axes[1].axhline(0,color='gray',linestyle=':');axes[1].set(xlabel='Article BF16 mean NLL',ylabel='Article mean NLL change',title='All 62 articles, seed-averaged');axes[1].legend()
    fig.suptitle('GPT-2 on DGX Spark: one p-bit per weight, T=32,768');fig.tight_layout();fig.savefig(dest/'full_benchmark.png',dpi=160);fig.savefig(dest/'full_benchmark.svg')
    lines=['# Completed GPT-2 benchmark','',f'BF16 perplexity: {baseline["PPL"]:.6f}. Finite-control mean: {cases[1]["PPL"]:.6f}.','',
        '| Method | Perplexity at mean NLL | Ratio to BF16 | 95% MC ratio interval | Frozen gate |','|---|---:|---:|---|---|']
    for kind in ['selected','matched_flat']:
        d=summary[kind];lo,hi=d['MC95_PPL_ratio_interval'];lines.append(f'| {kind} | {d["perplexity_at_mean_NLL"]:.6f} | {d["perplexity_ratio"]:.6f} | {lo:.6f}–{hi:.6f} | {d["passes_frozen_2pct_gate"]} |')
    lines+=['',report['scope'],'','A complete-cohort saved-output audit passed. Independent neural replay is a separate step; see its per-case receipts.','']
    (dest/'README.md').write_text('\n'.join(lines));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
