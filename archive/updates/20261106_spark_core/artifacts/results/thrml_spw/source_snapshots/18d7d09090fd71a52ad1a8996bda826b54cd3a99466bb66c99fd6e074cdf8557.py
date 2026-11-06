"""Accepted three-model summary requiring audits and independent neural replay."""
import json,math
from pathlib import Path
from .prepare_v1 import OUT,sha


def main():
 old=json.loads((OUT/'cross_model_summary_v1/results.json').read_text());lfm_audit=OUT/'lfm3_benchmark_audit_v1/results.json';lfm_replay=OUT/'lfm3_benchmark_replay_v1/results.json'
 a=json.loads(lfm_audit.read_text());replay=json.loads(lfm_replay.read_text());result_file=OUT/'lfm3_benchmark_v2/results.json'
 assert a['result_sha256']==replay['benchmark_result_sha256']==sha(result_file)
 assert len(replay['cases'])==7 and sum(len(x['replays'])for x in replay['cases'])==21
 assert all(y['losses_bitwise_equal']for x in replay['cases']for y in x['replays'])
 models=old['models']+[{'model':'LFM3-1.7B','seed_count':5,'perplexity_ratio':a['statistics']['PPL_ratio'],'MC95_perplexity_ratio_interval':a['statistics']['MC95_t4_PPL_ratio_interval'],'passes_own_frozen_quality_gates':a['statistics']['pass']}]
 sizes=[124439808,66955010,1728801536];tasks=['WikiText2:283225targets','IMDb:25000reviews','WikiText2:294442targets,own tokenizer']
 for model,size,task in zip(models,sizes,tasks):model.update({'unique_parameters':size,'task':task,'logical_pbits_per_learned_parameter':1,'samples_per_parameter':32768,'controller_bits':12})
 gpt=json.loads((OUT/'benchmark_v1/results.json').read_text());distil=json.loads((OUT/'distilbert_benchmark_v1/results.json').read_text());lfm=json.loads(result_file.read_text())
 for model,source in [(models[0],gpt),(models[2],lfm)]:
  model['BF16_PPL']=source['cases'][0]['PPL'];selected=[x['NLL']for x in source['cases']if x['case']['kind']=='selected'];model['selected_PPL_at_mean_NLL']=math.exp(sum(selected)/len(selected))
  model['seed_PPLs']=[math.exp(x)for x in selected]
 models[1]['quality_metric']='categorical perplexity and accuracy; not language-model perplexity'
 models[2]['all_parameter_count_hashes_replayed']=1430;models[2]['neural_windows_replayed']=21
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,ax=plt.subplots(figsize=(10,4.5),layout='constrained')
 labels=['GPT-2 · 124.4M parameters\nWikiText2 language modeling','DistilBERT · 67.0M parameters\nIMDb classification','LFM3 · 1.729B parameters\nWikiText2 language modeling']
 for i,m in enumerate(models):
  value=m['perplexity_ratio'];lo,hi=m['MC95_perplexity_ratio_interval'];ax.errorbar(value,i,xerr=[[value-lo],[hi-value]],fmt='o',capsize=5,color='#17836c')
 ax.axvline(1,color='#777777',linestyle=':');ax.axvline(1.02,color='#aa4444',linestyle='--',label='Prespecified 2% probability-quality margin')
 ax.set(yticks=range(3),yticklabels=labels,xlabel='Task perplexity / BF16 reference (five-seed 95% Monte Carlo interval)',title='One logical p-bit per learned parameter: three complete model/task checks')
 ax.set_ylim(2.4,-.4);ax.legend(loc='upper center',bbox_to_anchor=(.5,-.22),frameon=False);dest=OUT/'cross_model_summary_v2';dest.mkdir(exist_ok=True);fig.savefig(dest/'tested_model_quality.png',dpi=180);fig.savefig(dest/'tested_model_quality.svg');plt.close(fig)
 payload={'source_sha256':sha(__file__),'previous_two_model_summary_sha256':sha(OUT/'cross_model_summary_v1/results.json'),'LFM3_audit_sha256':sha(lfm_audit),'LFM3_replay_sha256':sha(lfm_replay),'models':models,'all_three_tested_models_pass':all(m['passes_own_frozen_quality_gates']for m in models),
 'scope':'Three checkpoint/task checks on two previously inspected benchmark tasks. Five independent selected-policy draws each; R1 ideal independent-binomial reconstruction, BF16 neural execution. All learned scalars K1 including bias/norms. Finite12bit controls and shared range metadata counted separately. Not universal numericBF16, physicalhardwarebenefit, fullmodelnativeTHRML, or unseen-task proof.'}
 (dest/'results.json').write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps({'all_three_pass':payload['all_three_tested_models_pass'],'models':[(m['model'],m['perplexity_ratio'])for m in models]}))
if __name__=='__main__':main()
