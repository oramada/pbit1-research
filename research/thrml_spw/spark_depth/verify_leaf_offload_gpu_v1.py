"""Small native LFM3 topology: streamed leaves versus resident BF16 CUDA."""
from pathlib import Path
import json,sys
from .prepare_v1 import OUT,sha
from .distilbert_benchmark_v1 import gpu_busy

def main():
    if gpu_busy():raise RuntimeError('Wait for the fixed GPT2 benchmark before CUDA offload tests')
    import torch
    from transformers import AutoConfig,AutoModelForCausalLM
    from .leaf_offload_v1 import LeafOffload
    torch.set_num_threads(2);torch.manual_seed(2026111601);torch.use_deterministic_algorithms(True)
    assert'GB10'in torch.cuda.get_device_name();torch.cuda.set_per_process_memory_fraction(min(1.,2*1024**3/torch.cuda.get_device_properties(0).total_memory))
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    cfg=AutoConfig.from_pretrained('data/thrml_spw/lfm3_1p7b/checkpoint',local_files_only=True,trust_remote_code=True)
    cfg.hidden_size=64;cfg.intermediate_size=128;cfg.num_hidden_layers=3;cfg.num_attention_heads=4;cfg.num_key_value_heads=2;cfg.head_dim=16;cfg.layer_types=['conv','full_attention','sliding_attention'];cfg.vocab_size=256;cfg.max_position_embeddings=1024;cfg.sliding_window=32;cfg.bos_token_id=0;cfg.eos_token_id=1;cfg.pad_token_id=2
    model=AutoModelForCausalLM.from_config(cfg,trust_remote_code=True,attn_implementation='eager').to(torch.bfloat16).eval()
    sys.modules[type(model.model.layers[0].conv).__module__].can_use_lfm3_conv=lambda *args:False
    root=OUT/'leaf_offload_gpu_verification_v1';root.mkdir(exist_ok=False);records=[]
    for batch,length in [(1,17),(2,65)]:
        ids=(torch.arange(batch*length).reshape(batch,length)%256).cuda();model.cuda()
        with torch.inference_mode():expected=model(input_ids=ids,use_cache=False).logits.float().cpu()
        model.cpu();torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode(),LeafOffload(model,'cuda')as offloader:
            actual=model(input_ids=ids,use_cache=False).logits.float().cpu();assert offloader.active_bytes==0
            assert all(p.device.type=='cpu'for p in model.parameters())
            assert model.lm_head.weight is model.model.embed_tokens.weight
            record={'batch':batch,'length':length,'logits_equal':bool(torch.equal(expected,actual)),'max_abs_difference':float((expected-actual).abs().max()),
                'peak_materialized_parameter_bytes':offloader.peak_active_bytes,'Torch_peak_GPU_allocated_bytes':torch.cuda.max_memory_allocated()}
        assert record['logits_equal'],record;records.append(record)
    (root/'results.json').write_text(json.dumps({'status':'tiny_LFM3_streamed_vs_resident_BF16_CUDA_bitwise_matches','source_sha256':sha(__file__),
        'offloader_sha256':sha(Path(__file__).with_name('leaf_offload_v1.py')),'records':records,
        'scope':'Three-layer random smallLFM3topology, allconv/full/slidingkinds. ValidatesCUDAleaftransfersemantics, notfull1.7Bqualityorfullmodelpeakmemory.'},indent=2)+'\n');print(json.dumps(records))

if __name__=='__main__':main()
