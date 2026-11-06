"""Rescore the cumulative study and independently replay six CPU forwards."""
from pathlib import Path
import json,hashlib,sys
import numpy as np
from .prepare_v1 import OUT,CHECKPOINT,sha
from .cpu_vector_validation_v1 import token_losses
sys.path.insert(0,str(Path('publish/pbit1/src').resolve()))
from pbit1 import FinitePBitProgram


def selected(name,condition):
    if condition=='all_parameters':return True
    parts=name.split('.')
    if parts[:2]!=['transformer','h']:return False
    index=int(parts[2])
    if condition=='all_blocks':return True
    n=int(condition[-1])
    return index<n if condition.startswith('prefix')else index>=12-n


def main():
    root=OUT/'gpt2_cumulative_v1';r=json.loads((root/'results.json').read_text());p=json.loads((root/'protocol.json').read_text());assert r['protocol_sha256']==sha(root/'protocol.json')
    assert len(r['cases'])==41 and len(p['seeds'])==5
    with np.load(OUT/'training_diagnostic_windows.npz')as f:inputs=f['input_ids'][:8,:129].copy()
    inventory=json.loads((OUT/'weight_survey_v1/results.json').read_text())['parameters'];summaries={}
    for case in r['cases']:
        label='baseline'if case['seed']is None else f"{case['condition']}_seed{case['seed']}";folder=root/label;losses=[]
        for row in range(8):
            with np.load(folder/f'row{row}.npz')as f:np.testing.assert_array_equal(f['input_ids'],inputs[row]);assert f['token_NLL'].shape==(128,);losses.extend(f['token_NLL'])
        np.testing.assert_allclose(np.mean(losses),case['NLL'],rtol=1e-14)
        expected=0 if case['seed']is None else sum(x['elements']for x in inventory if selected(x['name'],case['condition']))
        assert expected==case['converted_parameters']
        if case['condition'].startswith('suffix'):
            with np.load(folder/'channels.npz')as f:assert np.count_nonzero(f['error_SSE'][:12-int(case['condition'][-1])])==0
        summaries.setdefault(case['condition'],[]).append(case)
    assert summaries['baseline'][0]['point_sha256']==json.loads((OUT/'gaussian_control_v1/BF16/parameters.json').read_text())['point_sha256']
    for row in range(8):
        with np.load(root/'baseline'/f'row{row}.npz')as now,np.load(OUT/'gaussian_control_v1/BF16'/f'row{row}.npz')as before:
            np.testing.assert_array_equal(now['token_NLL'],before['token_NLL'])
    base=summaries['baseline'][0]['NLL'];curves={}
    for condition in p['conditions']:
        rows=summaries[condition];assert sorted(x['seed']for x in rows)==p['seeds'];d=np.array([x['NLL']for x in rows])-base;half=2.7764451051977987*d.std(ddof=1)/np.sqrt(5)
        curves[condition]={'delta_NLL':float(d.mean()),'MC95_delta_NLL':[float(d.mean()-half),float(d.mean()+half)],'converted_parameters':rows[0]['converted_parameters']}
    import torch
    from transformers import AutoModelForCausalLM
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);assert not torch.cuda.is_initialized()
    model=AutoModelForCausalLM.from_pretrained(CHECKPOINT,local_files_only=True,trust_remote_code=False,dtype=torch.float32,attn_implementation='eager').cpu().eval();params=dict(model.named_parameters());original={n:x.detach().to(torch.bfloat16).clone()for n,x in params.items()};draws={};seed=p['seeds'][0]
    records=json.loads((root/f'seed{seed}_draws.json').read_text())
    for index,(name,raw)in enumerate(original.items()):
        program=FinitePBitProgram.encode(raw.float().numpy(),group_size=128,layout=p['preferences'].get(name,'storage_flat'));sample=program.reconstruct(32768,seed=seed,parameter_index=index);draw=torch.from_numpy(sample.weights).to(torch.bfloat16)
        assert sample.counts_sha256==records[index]['counts_sha256']and hashlib.sha256(draw.view(torch.uint16).numpy().tobytes()).hexdigest()==records[index]['BF16_sha256'];draws[name]=draw
    replays=[]
    for condition in ['prefix6','suffix6','all_parameters']:
        point=hashlib.sha256()
        with torch.no_grad():
            for name,param in params.items():
                value=draws[name]if selected(name,condition)else original[name];param.copy_(value.float());point.update(name.encode());point.update(value.view(torch.uint16).numpy().tobytes())
        folder=root/f'{condition}_seed{seed}';assert point.hexdigest()==json.loads((folder/'results.json').read_text())['point_sha256']
        for row in [0,7]:
            with torch.inference_mode():out=model(input_ids=torch.from_numpy(inputs[row,None]),use_cache=False)
            loss=token_losses(out.logits.numpy(),inputs[row]);del out
            with np.load(folder/f'row{row}.npz')as f:np.testing.assert_array_equal(loss,f['token_NLL'])
            replays.append({'condition':condition,'row':row,'losses_bitwise_equal':True})
    assert not torch.cuda.is_initialized()
    result={'status':'all41_cumulative_conditions_audited_and_six_neural_rows_replayed','source_sha256':sha(__file__),'producer_result_sha256':sha(root/'results.json'),'curves':curves,'SDK_count_hashes_replayed':148,'replays':replays,'scope':p['scope']}
    (OUT/'gpt2_cumulative_v1_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(curves,indent=2))
if __name__=='__main__':main()
