"""All-learned-parameter finite programs from pinned BF16 checkpoint points.

No neural forward or evaluation token access. Bias and normalization scalars are
included; tied parameter identities receive exactly one canonical program.
"""
from pathlib import Path
import argparse,hashlib,json,time
import numpy as np
from .parameter_program_v1 import encode,resources

BASE=Path('results/thrml_spw/broad_verification_20261105')
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(16*1024*1024),b''):h.update(block)
    return h.hexdigest()

def main():
    import torch
    from transformers import AutoModelForCausalLM,AutoModelForSequenceClassification
    torch.set_num_threads(4)
    inventory_path=BASE/'additional_model_inventory/parameter_inventory.json'
    asset_path=BASE/'additional_assets_v1/manifest.json'
    data_audit_path=BASE/'additional_assets_v1_independent_data_audit.json'
    gpu_audit_path=BASE/'parameter_counts_h100_v1_audit.json'
    inventory=json.loads(inventory_path.read_text());assets=json.loads(asset_path.read_text())
    data_audit=json.loads(data_audit_path.read_text());gpu=json.loads(gpu_audit_path.read_text())
    assert data_audit['status']=='all_real_test_tokens_labels_article_targets_and_truncation_independently_rebuilt'
    assert sha(asset_path)==data_audit['manifest_sha256']
    assert gpu['status']=='all_synthetic_program_commands_tail_ranges_public_state_counts_readouts_INTcontrols_and_resources_independently_verified'
    assert gpu['public_state_bits_checked']==1376256
    out=BASE/'additional_parameter_packages_v1';out.mkdir(exist_ok=False)
    reports=[];begin=time.monotonic()
    for item in inventory['models']:
        name=item['model'];checkpoint=Path(assets['checkpoint_locations'][name])
        weights=checkpoint/'model.safetensors'
        expected=[v for k,v in assets['files'].items() if k.endswith('/'+str(weights)) or k==str(weights)]
        assert len(expected)==1 and sha(weights)==expected[0]['sha256']
        cls=AutoModelForCausalLM if item['model_type']=='gpt2' else AutoModelForSequenceClassification
        model=cls.from_pretrained(checkpoint,local_files_only=True,trust_remote_code=False,dtype=torch.bfloat16,
                                  attn_implementation='eager').eval()
        seen={};aliases={};canonical=[]
        for label,value in model.named_parameters(remove_duplicate=False):
            canonical_name=seen.setdefault(id(value),label);aliases[label]=canonical_name
            if canonical_name==label:canonical.append((label,value))
        assert aliases==item['aliases']
        actual=[{'name':name,'shape':list(p.shape),'elements':p.numel()} for name,p in canonical]
        assert actual==item['canonical_parameters']
        assert sum(p.numel() for _,p in canonical)==item['unique_learned_scalars']
        assert sum(p.numel() for name,p in canonical if name.endswith('.bias'))==item['bias_scalars']
        directory=out/item['model_type'];directory.mkdir()
        records=[]
        for index,(label,parameter) in enumerate(canonical):
            assert parameter.dtype==torch.bfloat16 and parameter.device.type=='cpu'
            point_sha=hashlib.sha256(parameter.detach().view(torch.uint16).numpy().tobytes()).hexdigest()
            program=encode(parameter.detach().float().numpy())
            path=directory/f'parameter_{index:04d}.npz'
            with path.open('xb') as f:np.savez_compressed(f,**program)
            records.append({'index':index,'name':label,'shape':list(parameter.shape),'elements':parameter.numel(),
                'original_BF16_raw_sha256':point_sha,'program':str(path),'program_sha256':sha(path),
                'program_serialized_bytes':path.stat().st_size,'constant_range_groups':int((program['span']==0).sum()),
                'allocation':resources(program,1,8192,65536)})
        buffers=[{'name':label,'shape':list(p.shape),'dtype':str(p.dtype),'elements':p.numel()} for label,p in model.named_buffers()]
        report={'status':'all_learned_BF16_checkpoint_parameters_encoded_without_neural_forward',
            'model':name,'revision':item['revision'],'checkpoint_sha256':sha(weights),'model_type':item['model_type'],
            'canonical_parameters':records,'aliases':aliases,'buffers_excluded_as_unlearned':buffers,
            'unique_learned_scalars':sum(p.numel() for _,p in canonical),'bias_scalars':item['bias_scalars'],
            'range_groups':sum(r['allocation']['range_groups'] for r in records),
            'range_metadata_bits':sum(r['allocation']['range_metadata_bits'] for r in records),
            'finite_controller_information_bits':12*item['unique_learned_scalars'],
            'one_logical_pbit_per_scalar':True,'all_constants_still_assigned_and_charged':True,
            'encoding_source_point':'Pinned checkpoint rounded once to BF16, then represented exactly as float32 for finite-command conversion. No task fitting.',
            'scope':'Full program conversion and identity coverage only. Complete independent package audit, neural quality/statistics and R2operator gates remain.'}
        path=directory/'package.json';path.write_text(json.dumps(report,indent=2)+'\n')
        reports.append({'model':name,'path':str(path),'sha256':sha(path),'parameters':len(records),
                        'unique_learned_scalars':report['unique_learned_scalars'],'range_groups':report['range_groups']})
        del model,canonical,program,parameter
    result={'status':'both_complete_parameter_packages_produced_pending_independent_audit',
        'packages':reports,'inputs_sha256':{str(p):sha(p) for p in [inventory_path,asset_path,data_audit_path,gpu_audit_path]},
        'sources_sha256':{str(p):sha(p) for p in [Path(__file__),Path(__file__).with_name('parameter_program_v1.py')]},
        'torch_version':torch.__version__,'numpy_version':np.__version__,'elapsed_seconds':time.monotonic()-begin,
        'scope':'No forward pass, evaluation token loading, quality outcome or full-model stochastic sampling.'}
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':main()
