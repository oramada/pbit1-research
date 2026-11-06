"""Training-only CPU activation diagnostics and range-direction selection.

Weights are rounded to BF16, but arithmetic is FP32 on CPU. This is a labelled
calibration approximation, not a BF16 or p-bit task-quality measurement.
"""
from pathlib import Path
import json,hashlib,time
import numpy as np
from .prepare_v1 import OUT,sha
from .weight_survey_v1 import mapped

def main():
    import torch
    from transformers import AutoModelForSequenceClassification
    torch.set_num_threads(2);assert not torch.cuda.is_initialized()
    prepared=OUT/'distilbert_preparation_v1';meta=json.loads((prepared/'results.json').read_text());assert sha(prepared/'calibration.npz')==meta['calibration_sha256']
    for file,digest in meta['files'].items():assert sha(file)==digest
    dest=OUT/'distilbert_cpu_diagnostics_v1';dest.mkdir(exist_ok=False);start=time.monotonic()
    protocol={'source_sha256':sha(__file__),'mapping_sha256':sha(Path(__file__).with_name('weight_survey_v1.py')),'prepared_sha256':sha(prepared/'results.json'),
        'reviews':64,'batch':2,'max_tokens':256,'precision':'BF16-rounded weights, FP32 CPU arithmetic','selector':'Minimize conditional sum of independent output variances using clean input second moments; embedding weighted by nonpadding lookup frequencies. Ignores downstream loss sensitivity.',
        'scope':'Training-only activation/moment/range diagnostics. No labels consumed, no pbit draws, no task-quality or heldout claim. No GPU.'}
    (dest/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    model=AutoModelForSequenceClassification.from_pretrained(meta['checkpoint'],local_files_only=True,trust_remote_code=False,dtype=torch.float32).cpu().eval()
    with torch.no_grad():
        for p in model.parameters():p.copy_(p.to(torch.bfloat16).to(torch.float32))
    modules={n:m for n,m in model.named_modules()if isinstance(m,(torch.nn.Linear,torch.nn.Embedding,torch.nn.LayerNorm))};stats={};handles=[];mask=None
    def hook(name,module,args,output):
        x=args[0].detach().cpu().numpy();y=output.detach().cpu().numpy()
        if y.ndim==3:
            shape=(*mask.shape,y.shape[-1]);z=np.broadcast_to(y,shape)[mask];weights=np.broadcast_to(x,mask.shape)[mask]if isinstance(module,torch.nn.Embedding)else np.broadcast_to(x,(*mask.shape,x.shape[-1]))[mask]
        else:z=y.reshape(-1,y.shape[-1]);weights=x.reshape(-1,x.shape[-1])
        z=z.astype(np.float64);state=stats.setdefault(name,{'count':0,'output_sum':np.zeros(z.shape[-1]),'output_sumsq':np.zeros(z.shape[-1]),'output_maxabs':np.zeros(z.shape[-1])})
        state['count']+=len(z);state['output_sum']+=z.sum(0);state['output_sumsq']+=(z*z).sum(0);state['output_maxabs']=np.maximum(state['output_maxabs'],np.abs(z).max(0))
        if isinstance(module,torch.nn.Embedding):
            state.setdefault('lookup_counts',np.zeros(module.num_embeddings,np.int64));state['lookup_counts']+=np.bincount(weights.reshape(-1),minlength=module.num_embeddings)
        else:
            a=weights.astype(np.float64);state.setdefault('input_sumsq',np.zeros(a.shape[-1]));state['input_sumsq']+=(a*a).sum(0)
    for name,module in modules.items():handles.append(module.register_forward_hook(lambda m,a,o,name=name:hook(name,m,a,o)))
    with np.load(prepared/'calibration.npz',allow_pickle=False)as f:ids=f['input_ids'].copy();masks=f['attention_mask'].copy()
    assert ids.shape==(64,256)
    try:
        for offset in range(0,64,2):
            mask=masks[offset:offset+2].astype(bool)
            with torch.inference_mode():model(input_ids=torch.from_numpy(ids[offset:offset+2]),attention_mask=torch.from_numpy(masks[offset:offset+2]))
            (dest/'progress.json').write_text(json.dumps({'stage':'CPU_forwards','reviews':offset+2,'elapsed_seconds':time.monotonic()-start})+'\n')
    finally:
        for h in handles:h.remove()
    arrays={};report=[];preferences={}
    for name,module in modules.items():
        state=stats[name];n=state['count'];weight=module.weight.detach().numpy();entry={'module':name,'type':type(module).__name__,'observations':n,'shape':list(weight.shape),
            'output_RMS':float(np.sqrt(state['output_sumsq'].sum()/n/len(state['output_sumsq']))),'largest_output_channels':np.argsort(state['output_sumsq'])[-8:][::-1].tolist()}
        for key,array in state.items():
            if isinstance(array,np.ndarray):arrays[name+'.'+key]=array
        if isinstance(module,(torch.nn.Linear,torch.nn.Embedding)):
            variants={}
            for orientation in ['storage_flat','transposed_matrix']:
                oriented=weight if orientation=='storage_flat'else weight.T.copy();codes,lo,span,mean,variance=mapped(oriented,128);variance=variance.reshape(oriented.shape)
                if orientation=='transposed_matrix':variance=variance.T
                if isinstance(module,torch.nn.Linear):contribution=variance.sum(0)*(state['input_sumsq']/n)
                else:contribution=variance.sum(1)*(state['lookup_counts']/n)
                noise=float(contribution.sum()/8192);variants[orientation]={'predicted_sum_output_variance_T8192':noise,
                    'predicted_relative_output_RMSE_T8192':float(np.sqrt(noise/(state['output_sumsq'].sum()/n))),'top_contributing_inputs':np.argsort(contribution)[-8:][::-1].tolist(),
                    'range_groups':len(lo),'control_range_state_bits':13*weight.size+64*len(lo)}
            best=min(variants,key=lambda k:variants[k]['predicted_sum_output_variance_T8192']);entry['variants']=variants;entry['preference']=best;preferences[name+'.weight']=best
        report.append(entry);(dest/'progress.json').write_text(json.dumps({'stage':'range_analysis','module':name,'elapsed_seconds':time.monotonic()-start})+'\n')
    assert not torch.cuda.is_initialized()
    with(dest/'channel_statistics.npz').open('xb')as f:np.savez_compressed(f,**arrays)
    result={'status':'CPU_training_diagnostics_and_operator_preferences_complete','protocol_sha256':sha(dest/'protocol.json'),'modules':report,'preferences':preferences,
        'canonical_parameters':len(list(model.parameters())),'unique_parameter_count':sum(p.numel()for p in model.parameters()),'channel_statistics_sha256':sha(dest/'channel_statistics.npz'),
        'scope':protocol['scope'],'elapsed_seconds':time.monotonic()-start}
    (dest/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'modules':len(report),'selected_weights':len(preferences),'elapsed_seconds':time.monotonic()-start}),flush=True)

if __name__=='__main__':main()
