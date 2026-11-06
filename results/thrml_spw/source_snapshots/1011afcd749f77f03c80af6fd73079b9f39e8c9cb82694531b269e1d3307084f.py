"""Tiny LFM3 CPU-copy tests; CUDA and full-checkpoint validation still required."""
from pathlib import Path
import json,sys
from .prepare_v1 import OUT,sha

def main():
    import torch
    from transformers import AutoConfig,AutoModelForCausalLM
    from .leaf_offload_v1 import LeafOffload
    torch.set_num_threads(2);torch.manual_seed(2026111601);assert not torch.cuda.is_available()
    path='data/thrml_spw/lfm3_1p7b/checkpoint';cfg=AutoConfig.from_pretrained(path,local_files_only=True,trust_remote_code=True)
    cfg.hidden_size=64;cfg.intermediate_size=128;cfg.num_hidden_layers=3;cfg.num_attention_heads=4;cfg.num_key_value_heads=2;cfg.head_dim=16
    cfg.layer_types=['conv','full_attention','sliding_attention'];cfg.vocab_size=256;cfg.max_position_embeddings=1024;cfg.sliding_window=32;cfg.bos_token_id=0;cfg.eos_token_id=1;cfg.pad_token_id=2
    model=AutoModelForCausalLM.from_config(cfg,trust_remote_code=True,attn_implementation='eager').cpu().eval()
    sys.modules[type(model.model.layers[0].conv).__module__].can_use_lfm3_conv=lambda *args:False
    ids=torch.arange(66).reshape(2,33)%256;before={n:p.detach().clone()for n,p in model.named_parameters()}
    with torch.inference_mode():reference=model(input_ids=ids,use_cache=False).logits.clone()
    with torch.inference_mode(),LeafOffload(model,'cpu',cpu_copy_test=True)as offload:
        actual=model(input_ids=ids,use_cache=False).logits
        assert torch.equal(actual,reference);assert offload.active_bytes==0 and all(n==0 for n in offload.refs.values());calls=list(offload.calls);peak=offload.peak_active_bytes
    assert model.lm_head.weight is model.model.embed_tokens.weight
    for name,p in model.named_parameters():assert torch.equal(p,before[name])
    exception_cleaned=False
    try:
        with torch.inference_mode(),LeafOffload(model,'cpu',cpu_copy_test=True):
            def fail(module,args):raise RuntimeError('intentional test fault')
            hook=model.lm_head.register_forward_pre_hook(fail)
            try:model(input_ids=ids,use_cache=False)
            finally:hook.remove()
    except RuntimeError as error:
        assert str(error)=='intentional test fault';exception_cleaned=True
    assert exception_cleaned and model.lm_head.weight is model.model.embed_tokens.weight
    for name,p in model.named_parameters():assert p.device.type=='cpu'and torch.equal(p,before[name])
    assert not torch.cuda.is_initialized()
    root=OUT/'leaf_offload_cpu_verification_v1';root.mkdir(exist_ok=False)
    (root/'results.json').write_text(json.dumps({'status':'tiny_LFM3_CPU_copy_alias_output_and_exception_cleanup_checks_passed','source_sha256':sha(__file__),'offloader_sha256':sha(Path(__file__).with_name('leaf_offload_v1.py')),
        'parameter_count':sum(p.numel()for p in model.parameters()),'parameterized_calls':calls,'peak_materialized_weight_bytes_CPU_test':peak,'exception_cleanup':exception_cleaned,
        'scope':'Synthetic3layerLFM3,CPUclone-materializationonly. NoCUDA,fullcheckpoint,qualityor2GiBpeakclaim.'},indent=2)+'\n');print('Tiny CPU-copy LFM3 offload tests passed; full CUDA validation remains.')

if __name__=='__main__':main()
