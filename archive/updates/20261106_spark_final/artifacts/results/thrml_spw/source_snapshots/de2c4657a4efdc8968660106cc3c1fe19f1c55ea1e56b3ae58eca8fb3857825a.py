"""Pinned eager BF16 model execution, exact aliases and fixed input batches."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import json
import numpy as np
import torch
from .r1_io_v1 import sha,array_sha,program
from .parameter_program_v1 import expectation,integer_control


def load_model(spec):
    from transformers import AutoModelForCausalLM,AutoModelForSequenceClassification
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    assert torch.cuda.is_available() and 'H100' in torch.cuda.get_device_name()
    assert sha(spec['package'])==spec['package_sha256']
    package=json.loads(Path(spec['package']).read_text())
    checkpoint=Path(spec['checkpoint']);assert sha(checkpoint/'model.safetensors')==package['checkpoint_sha256']
    cls=AutoModelForCausalLM if package['model_type']=='gpt2' else AutoModelForSequenceClassification
    model=cls.from_pretrained(checkpoint,local_files_only=True,trust_remote_code=False,dtype=torch.bfloat16,
                              attn_implementation='eager').cuda().eval()
    aliases={};seen={};parameters={}
    for name,p in model.named_parameters(remove_duplicate=False):
        aliases[name]=seen.setdefault(id(p),name)
        if aliases[name]==name:parameters[name]=p
    assert aliases==package['aliases']
    assert list(parameters)==[v['name'] for v in package['canonical_parameters']]
    for r in package['canonical_parameters']:
        p=parameters[r['name']]
        assert list(p.shape)==r['shape'] and p.dtype==torch.bfloat16
        import hashlib
        assert hashlib.sha256(p.detach().cpu().view(torch.uint16).numpy().tobytes()).hexdigest()==r['original_BF16_raw_sha256']
    return model,parameters,package


def install(parameters,name,values):
    p=parameters[name];value=np.asarray(values)
    assert value.dtype==np.float32 and value.shape==tuple(p.shape) and np.isfinite(value).all()
    # The two specified casts happen on CPU, before transfer to the GPU.
    rounded=torch.from_numpy(value.copy()).to(torch.bfloat16)
    digest=array_sha(rounded.view(torch.uint16).numpy())
    p.data=rounded.to(device=p.device)
    return digest


def control(parameters,record,arm):
    if arm=='original':return array_sha(parameters[record['name']].detach().cpu().view(torch.uint16).numpy())
    p=program(record['program'])
    if arm.startswith('mean'):value=expectation(p,int(arm[4:])).astype(np.float32)
    elif arm.startswith('int'):
        original=parameters[record['name']].detach().float().cpu().numpy()
        value=integer_control(original,int(arm[3:])).astype(np.float32)
    else:raise ValueError('Unknown deterministic control')
    return install(parameters,record['name'],value)


def batches(spec):
    data=spec['data']
    if data['kind']=='synthetic':
        with np.load(data['arrays'],allow_pickle=False) as a:
            yield {'index':0,'task':spec['model_type'],'synthetic':True},a['input_ids'].copy(),a['attention_mask'].copy() if 'attention_mask' in a else None,None
        return
    if spec['model_type']=='gpt2':
        layout=json.loads(Path(data['layout']).read_text())
        with np.load(data['arrays'],allow_pickle=False) as a:tokens,offsets=a['tokens'].copy(),a['offsets'].copy()
        for index,w in enumerate(layout['windows']):
            begin=int(offsets[w['article']]);ids=tokens[begin+w['context_start']:begin+w['context_stop']][None]
            yield {'index':index,'task':'gpt2',**w},ids,None,None
    else:
        with np.load(data['arrays'],allow_pickle=False) as a:ids,mask,labels=a['input_ids'].copy(),a['attention_mask'].copy(),a['labels'].copy()
        for begin in range(0,len(ids),128):
            end=min(begin+128,len(ids))
            yield {'index':begin//128,'task':'distilbert','review_start':begin,'review_stop':end},ids[begin:end],mask[begin:end],labels[begin:end]


def forward(model,ids,mask):
    kwargs={'input_ids':torch.from_numpy(ids).cuda()}
    if mask is not None:kwargs['attention_mask']=torch.from_numpy(mask).cuda()
    if model.config.model_type=='gpt2':kwargs['use_cache']=False
    with torch.inference_mode():logits=model(**kwargs).logits.float().cpu().numpy()
    assert logits.dtype==np.float32 and np.isfinite(logits).all()
    return logits


def observations(record,ids,labels,logits):
    if record.get('synthetic'):return {'logits':logits}
    if record['task']=='distilbert':
        return {'logits':logits,'labels':labels,'correct':(logits.argmax(-1)==labels).astype(np.int8)}
    first=record['target_start']-record['context_start'];last=record['target_stop']-record['context_start']
    assert 1<=first<last<=ids.shape[1]
    n=last-first;loss=np.empty(n,np.float64);normalizer=np.empty_like(loss);target_logit=np.empty_like(loss)
    for start in range(first,last,16):
        stop=min(start+16,last);z=torch.from_numpy(logits[0,start-1:stop-1].copy()).double()
        targets=ids[0,start:stop];rows=torch.arange(stop-start)
        normal=torch.logsumexp(z,dim=-1).numpy();selected=z[rows,torch.from_numpy(targets)].numpy()
        a,b=start-first,stop-first;normalizer[a:b]=normal;target_logit[a:b]=selected;loss[a:b]=normal-selected
    assert np.isfinite(loss).all()
    return {'token_NLL':loss,'log_normalizer':normalizer,'target_logit':target_logit,'target_ids':ids[0,first:last]}
