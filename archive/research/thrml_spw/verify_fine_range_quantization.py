"""Independent original-model quantization and raw numerical/resource verification."""
import csv,json,hashlib
from pathlib import Path
from decimal import Decimal,localcontext
import numpy as np
import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM
from huggingface_hub.constants import HF_HUB_CACHE
OUT=Path('results/thrml_spw/fine_range_quantization_controls')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def close(a,b):np.testing.assert_allclose(a,b,rtol=1e-12,atol=1e-12)
def reconstruct(weight,variant,controls,book):
    if variant['kind']=='FP32':return weight.copy()
    g=variant['group'];out=np.empty_like(weight)
    for first in range(0,len(weight),g):
        w=weight[first:first+g]
        if variant['kind']=='programmed_mean':
            group=first//g;lo=controls['lower'][group].astype(float);span=controls['span'][group].astype(float);out[first:first+g]=(lo+span*book[controls['codes'][first:first+g]]).astype(np.float32);continue
        bits=variant['bits'];method=variant['method']
        if method=='symmetric':
            maximum=(1<<(bits-1))-1;scale=np.max(np.abs(w),axis=0)/maximum;safe=np.where(scale==0,np.float32(1),scale);out[first:first+g]=np.clip(np.rint(w/safe),-maximum,maximum)*scale
        else:
            maximum=(1<<bits)-1;lo=w.min(0);hi=w.max(0)
            if method=='affine_zero':lo=np.minimum(lo,0);hi=np.maximum(hi,0)
            scale=(hi-lo)/maximum;safe=np.where(scale==0,np.float32(1),scale)
            if method=='minmax':code=np.clip(np.rint((w-lo)/safe),0,maximum);out[first:first+g]=lo+code*scale
            else:zero=np.clip(np.rint(-lo/safe),0,maximum);code=np.clip(np.rint(w/safe)+zero,0,maximum);out[first:first+g]=(code-zero)*scale
    return out
def main():
    torch.set_num_threads(8);torch.use_deterministic_algorithms(True);p=json.loads((OUT/'protocol.json').read_text());m=json.loads((OUT/'manifest.json').read_text());assert m['protocol_sha256']==sha(OUT/'protocol.json') and m['raw_tokens_sha256']==sha(OUT/'raw_tokens.npz') and m['weight_hashes_sha256']==sha(OUT/'weight_hashes.json') and m['variants']==21 and m['stochastic_samples_generated']==0
    for name,h in p['source_sha256'].items():assert sha(name)==h
    assert sha(p['source_tokens'])==p['tokens_sha256'] and sha(OUT/'probability_codebook.npy')==p['probability_codebook_sha256'];book=np.load(OUT/'probability_codebook.npy')
    with localcontext() as ctx:
        ctx.prec=60
        for code,q in enumerate(book):
            field=np.float32(-4+code*8/255);d=Decimal(float(field));exact=float(1/(1+(-2*d).exp()));np.testing.assert_allclose(q,exact,rtol=3e-16,atol=2e-16)
    model=AutoModelForCausalLM.from_pretrained(Path(HF_HUB_CACHE)/f"models--distilbert--distilgpt2/snapshots/{p['revision']}",local_files_only=True,attn_implementation='eager').float().eval();modules=[model.get_submodule(name) for name in p['targets_in_forward_order']];original=[mod.weight.detach().clone() for mod in modules];weights=[a.numpy().copy() if mod.__class__.__name__=='Conv1D' else a.numpy().T.copy() for a,mod in zip(original,modules)];biases=sum(mod.bias.numel() for mod in modules);n=sum(w.size for w in weights);assert n==p['converted_weights'] and len({id(v) for v in modules})==24;tokens=np.load(p['source_tokens'])['test_tokens'];truth=tokens[:,1:];raw=np.load(OUT/'raw_tokens.npz');records=json.loads((OUT/'weight_hashes.json').read_text());metrics=list(csv.DictReader((OUT/'metrics.csv').open()));checked=0
    for variant in p['variants']:
        expected=next(r for r in records if r['condition']==variant['name']);cfg=[]
        try:
            for stage,(mod,w) in enumerate(zip(modules,weights)):
                controls=None
                if variant['kind']=='programmed_mean':
                    c=p['control_artifacts'][str(variant['group'])][stage];assert sha(c['path'])==c['sha256'];controls=np.load(c['path'])
                q=reconstruct(w,variant,controls,book);assert q.shape==w.shape and q.dtype==np.float32 and digest(q)==expected['weights'][stage]['weight_sha256'];checked+=w.size
                with torch.no_grad():mod.weight.copy_(torch.from_numpy(q if mod.__class__.__name__=='Conv1D' else q.T))
            nll=[];pred=[]
            with torch.inference_mode():
                for row in tokens:
                    ids=torch.from_numpy(row[None]);logits=model(input_ids=ids,use_cache=False).logits[:,:-1];nll.append(F.cross_entropy(logits.reshape(-1,logits.shape[-1]),ids[:,1:].reshape(-1),reduction='none').numpy());pred.append(logits.argmax(-1).numpy()[0])
            nll,pred=np.stack(nll),np.stack(pred);np.testing.assert_array_equal(nll,raw[variant['name']+'_nll']);np.testing.assert_array_equal(pred,raw[variant['name']+'_predictions']);ce=float(nll.astype(float).mean());row=next(r for r in metrics if r['condition']==variant['name']);stored=next(r for r in m['scores'] if r['condition']==variant['name'])
            for k,v in {'cross_entropy':ce,'perplexity':np.exp(ce),'token_accuracy':np.mean(pred==truth)}.items():close(float(row[k]),v);close(stored[k],v)
            if variant['kind']=='FP32':codebits=32*n;metabits=0;groups=0
            else:
                groups=n//variant['group'];codebits=(8 if variant['kind']=='programmed_mean' else variant['bits'])*n;metabits=groups*(64 if variant['kind']=='programmed_mean' or variant.get('method')=='minmax' else (32 if variant['method']=='symmetric' else 32+variant['bits']))
            payload=(codebits+metabits+32*biases)//8;flags=48 if variant.get('group')==16 else 0;lookup=2048 if variant['kind']=='programmed_mean' else 0;outside=p['model_unique_parameters']-n-biases
            for k,v in {'converted_weight_code_bits':codebits,'range_metadata_bits':metabits,'affine_bias_bits':32*biases,'group_count':groups,'code_range_bias_payload_bytes':payload,'configuration_bytes':flags,'global_lookup_bytes':lookup,'bits_per_converted_weight_including_affine_bias':8*payload/n,'remaining_FP32_parameters_outside_payload':outside,'mixed_parameter_payload_bytes':payload+flags+lookup+4*outside,'stochastic_samples_generated':0}.items():close(float(row[k]),v);close(stored[k],v)
            print(f'Independent original-model control {variant["name"]}: all weight hashes and token metrics exact',flush=True)
        finally:
            with torch.no_grad():
                for mod,w in zip(modules,original):mod.weight.copy_(w)
    assert checked==21*n;r={'passed':True,'variants':21,'weight_coordinates_independently_reconstructed':checked,'original_model_token_losses_predictions_exact':True,'all_score_and_payload_rows_verified':True,'probability_codebook_checked_against_60digit_decimal':256,'source_sha256':sha(__file__),'scope':'Independent deterministic quantizers, original-model replay and raw score/resource audit on consumed diagnostic text. No native sampling, new confirmation or optimized integer-kernel hardware claim.'};(OUT/'evidence_audit.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
if __name__=='__main__':main()
