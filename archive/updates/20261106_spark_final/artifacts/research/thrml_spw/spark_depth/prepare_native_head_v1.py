"""Prepare actual GPT2 head weights and finite controls for native THRML audit."""
from pathlib import Path
import json
import numpy as np
from .prepare_v1 import OUT,CHECKPOINT,sha
from .weight_survey_v1 import mapped

def subset_encoding(weights,tokens,layout):
    V,D=weights.shape;source=weights.T.copy().reshape(-1)if layout=='transposed_matrix'else weights.reshape(-1)
    indices=np.arange(D)[None,:]*V+tokens[:,None]if layout=='transposed_matrix'else tokens[:,None]*D+np.arange(D)[None,:]
    groups=indices//128;unique,reverse=np.unique(groups,return_inverse=True);positions=unique[:,None]*128+np.arange(128)[None,:]
    data=source[np.minimum(positions,len(source)-1)].copy();data[positions>=len(source)]=np.nan
    codes,lo,span,mean,var=mapped(data.reshape(-1),128)
    loc=reverse.reshape(indices.shape)*128+(indices%128)
    return codes[loc],np.repeat(lo,128)[loc],np.repeat(span,128)[loc],mean[loc],var[loc]

def main():
    import torch
    from safetensors.torch import load_file
    torch.set_num_threads(2)
    root=OUT/'native_head_v1';root.mkdir(exist_ok=False)
    state=load_file(str(CHECKPOINT/'model.safetensors'),device='cpu');w=state['wte.weight'].to(torch.bfloat16).float().numpy();assert w.shape==(50257,768)
    with np.load(OUT/'training_diagnostic_windows.npz')as d:tokens=d['input_ids'][:8,1].copy()
    assert len(np.unique(tokens))==8
    hidden=[]
    for row in range(8):
        with np.load(OUT/'trace_v1/original_BF16'/f'row{row}.npz')as f:hidden.append(f['transformer.ln_f'][0,0].copy())
    arrays={'hidden':np.array(hidden),'target_token_ids':tokens,'original_weights':w[tokens]}
    for layout in ['storage_flat','transposed_matrix']:
        codes,lo,span,mean,var=subset_encoding(w,tokens,layout)
        arrays.update({layout+'_codes':codes,layout+'_lower':lo,layout+'_span':span,layout+'_mean':mean,layout+'_variance':var})
    with(root/'inputs.npz').open('xb')as f:np.savez_compressed(f,**arrays)
    protocol={'prepare_source_sha256':sha(__file__),'inputs_sha256':sha(root/'inputs.npz'),'checkpoint_sha256':sha(CHECKPOINT/'model.safetensors'),
        'native_source_sha256':sha(Path('research/thrml_spw/native.py')),'layouts':['storage_flat','transposed_matrix'],'seeds':list(range(2026110701,2026110706)),
        'samples':8192,'nested_budgets':[128,512,2048,8192],'n_pbits':6144,'mode':'OfficialTHRML unary SpinNode factors, CPU backend, no NumPy replacement sampler.',
        'scope':'Isolated real8x768outputweightbank and8clean hiddenvectors fromtrainingdiagnostics. Samefinitecontrolfields; noparameterfitting. Fiveindependentnativechains perlayout. Doesnotconstitutefullmodelnativevalidation.',
        'weight_moment_gate':'Everylayout/seed/prefix independently computed weightSSE versusexactfinitecontrolmean within8standarddeviations of independentBernoulliSSE expectation, before any output projection. No appqualitygate.'}
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');print(json.dumps({'prepared_pbits':6144,'arrays_sha256':protocol['inputs_sha256']}))

if __name__=='__main__':main()
