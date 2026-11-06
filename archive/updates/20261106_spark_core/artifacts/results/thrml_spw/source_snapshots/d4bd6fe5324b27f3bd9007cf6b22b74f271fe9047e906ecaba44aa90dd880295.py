"""Freeze a cross-architecture evaluation without executing its test forwards."""
from pathlib import Path
import json,math
import numpy as np
from .prepare_v1 import OUT,sha

def main():
    parent=Path('results/thrml_spw/broad_verification_20261105/additional_assets_v1_mirror/results/thrml_spw/broad_verification_20261105/additional_assets_v1')
    arrays=parent/'distilbert_tokens.npz';layout_file=parent/'distilbert_layout.json';layout=json.loads(layout_file.read_text())
    with np.load(arrays,allow_pickle=False)as a:
        ids=a['input_ids'];mask=a['attention_mask'];labels=a['labels']
        assert ids.shape==mask.shape==(25000,256)and labels.shape==(25000,)
        assert set(np.unique(labels))=={0,1}and np.all(mask.sum(1)>0)
        label_counts=np.bincount(labels,minlength=2).tolist()
    prep=OUT/'distilbert_preparation_v1/results.json';diagnostics=OUT/'distilbert_cpu_diagnostics_v1/results.json';selection=OUT/'profile_extension_combined_v1.json'
    selected=json.loads(selection.read_text());assert selected['selected_profile']=='operator_all_T32768'
    preferences=json.loads(diagnostics.read_text())['preferences'];assert len(preferences)==40
    out=OUT/'distilbert_eval_plan_v1.json'
    protocol={'status':'cross_architecture_policy_and_full_data_frozen_no_test_forward','source_sha256':sha(__file__),
        'model_preparation':str(prep),'model_preparation_sha256':sha(prep),'training_diagnostics_sha256':sha(diagnostics),'GPT2_calibration_selection_sha256':sha(selection),
        'model':'distilbert/distilbert-base-uncased-finetuned-sst-2-english','unique_parameters':66955010,
        'arrays':str(arrays),'arrays_sha256':sha(arrays),'layout':str(layout_file),'layout_sha256':sha(layout_file),'reviews':25000,'label_counts':label_counts,'input_shape':[25000,256],
        'group_size':128,'control_bits':12,'field_range':[-4,4],'samples':32768,'preferences':preferences,
        'cases':[{'name':'BF16','kind':'original'},{'name':'selected_mean','kind':'mean'}]+[{'name':f'{kind}_seed{seed}','kind':kind,'seed':seed}for seed in range(2026111001,2026111006)for kind in ['selected','matched_flat']],
        'batch_size':32,'GPU_allocator_cap_bytes':2*1024**3,
        'accuracy_gate':'Five-seed mean accuracy decline plus t4(97.5%) standarderror <=.02 absolute; reference computed locally in BF16 onidenticalreviews.',
        'probability_gate':'Five-seed mean delta crossentropy plus t4(97.5%) standarderror <= ln(1.02). This categoricalperplexity gate matches GPT2 distribution-level tolerance.',
        'joint_rule':'Report accuracy and probability gates separately; call cross-architecture jointquality success only if both andGPT2primary pass. Do notreinterpretolderaccuracy-onlypass.',
        'scope':'SameT32768 andK1asGPT2selectedpolicy; peroperatordirectionselectedfrom64trainingreviews, nottestlabels. FullIMDbtestwasalreadyusedinearlierresearch: regressionre-evaluation, notnewunexposeddata. No GPUjobstartedbythispreparation.'}
    with out.open('x')as f:json.dump(protocol,f,indent=2);f.write('\n')
    print(json.dumps({'frozen_reviews':25000,'frozen_cases':12,'label_counts':label_counts}))

if __name__=='__main__':main()
