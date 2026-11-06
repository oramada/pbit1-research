"""Pin canonical learned parameter shapes without weights, data or scores."""
from pathlib import Path
import hashlib
import json
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSequenceClassification


def main():
    root=Path('results/thrml_spw/broad_verification_20261105/additional_model_inventory')
    manifest=json.loads((root/'pinned_hub_inventory.json').read_text())
    records=[]
    for item in manifest['records']:
        if item['kind']!='model':continue
        config=AutoConfig.from_pretrained(item['id'],revision=item['revision'],trust_remote_code=False)
        loader=AutoModelForCausalLM if item['id'].endswith('/gpt2') else AutoModelForSequenceClassification
        with torch.device('meta'):
            model=loader.from_config(config,attn_implementation='eager',trust_remote_code=False)
        model.tie_weights()
        canonical={};aliases={};parameters=[]
        for name,param in model.named_parameters(remove_duplicate=False):
            # Meta tensors all have address0; Python parameter identity is
            # essential for distinguishing real ties from unrelated tensors.
            if id(param) not in canonical:
                canonical[id(param)]=name
                parameters.append({'name':name,'shape':list(param.shape),'elements':param.numel()})
            aliases[name]=canonical[id(param)]
        total=sum(p['elements'] for p in parameters)
        assert total==sum(p.numel() for p in model.parameters())
        records.append({'model':item['id'],'revision':item['revision'],'model_type':config.model_type,
            'canonical_parameters':parameters,'aliases':aliases,'unique_learned_scalars':total,
            'bias_scalars':sum(p['elements'] for p in parameters if p['name'].endswith('.bias')),
            'one_pbit_assignments_if_all_learned_scalars_converted':total,
            'scope':'Metadata-only architecture inventory. Biases and learned normalization parameters are included. No pretrained weights loaded and no examples or task scores evaluated.'})
    result={'status':'additional_decoder_and_encoder_parameter_inventories_pinned_before_scores',
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'hub_inventory_sha256':hashlib.sha256((root/'pinned_hub_inventory.json').read_bytes()).hexdigest(),
        'torch_version':torch.__version__,'models':records}
    with (root/'parameter_inventory.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps([{'model':r['model'],'unique_learned_scalars':r['unique_learned_scalars'],'bias_scalars':r['bias_scalars']} for r in records]))


if __name__=='__main__':main()
