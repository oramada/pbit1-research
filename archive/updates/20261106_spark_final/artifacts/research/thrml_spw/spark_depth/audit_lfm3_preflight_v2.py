"""Check full-checkpoint identity and synthetic offload outputs independently."""
from pathlib import Path
import hashlib,json
import numpy as np
from .prepare_v1 import OUT,sha

def main():
    import torch
    from safetensors import safe_open
    root=OUT/'lfm3_full_preflight_v2';r=json.loads((root/'results.json').read_text());protocol=json.loads((root/'protocol.json').read_text());assert r['protocol_sha256']==sha(root/'protocol.json')
    canonical=json.loads((OUT/'lfm3_memory_plan_v1/results.json').read_text())['canonical'];h=hashlib.sha256();total=0
    with safe_open('data/thrml_spw/lfm3_1p7b/checkpoint/model.safetensors',framework='pt',device='cpu')as f:
        for p in canonical:
            value=f.get_tensor(p['name']);assert list(value.shape)==p['shape']and value.dtype==torch.bfloat16
            h.update(p['name'].encode());h.update(value.view(torch.uint16).numpy().tobytes());total+=value.numel()
    assert total==1728801536;point=h.hexdigest()
    for i,case in enumerate(r['cases']):
        file=root/f'case{i}_logits.npz';assert sha(file)==case['file_sha256']and case['weight_point_sha256']==point
        assert case['Torch_GPU_peak_allocated_bytes']<=2*1024**3 and case['parameterized_module_calls']==287
        with np.load(file,allow_pickle=False)as f:
            ids=f['input_ids'];logits=f['logits'];expected=(np.arange(ids.size).reshape(ids.shape)*7919+17)%128000
            assert np.array_equal(ids,expected)and logits.shape==tuple(case['logits_shape'])
            assert np.isfinite(logits).all()and np.all((logits.view(np.uint32)&65535)==0)
    assert not torch.cuda.is_initialized()
    result={'status':'original_checkpoint_hash_outputs_module_coverage_and_peak_allocation_verified','source_sha256':sha(__file__),'preflight_sha256':sha(root/'results.json'),
        'weight_point_sha256':point,'parameters':total,'maximum_Torch_GPU_allocation_bytes':max(c['Torch_GPU_peak_allocated_bytes']for c in r['cases']),
        'scope':'CPUauditofpreflightartifacts. FullmodelBF16executionpath, noonepbitreplacementortaskqualityevidence.'}
    (OUT/'lfm3_full_preflight_v2_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':main()
