"""Exercise actual LFM3 forward geometries using synthetic IDs, without scores."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import sys
import numpy as np
import torch
from transformers import AutoModelForCausalLM
from .program_export import load_program
from .holdout_frontend_v2 import prepare, restore


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    root = Path('results/thrml_spw/lfm3_1p7b')
    out = Path('results/thrml_spw/broad_verification_20261105/model_geometry_v3')
    out.mkdir(parents=True, exist_ok=False)
    aliases = json.loads((root/'adaptation_pilot/mean/module_aliases.json').read_text())
    package = json.loads((root/'adaptation_pilot/mean/program_package.json').read_text())
    records = {r['name']: r for r in package['programs']}
    torch.set_num_threads(4)
    model = AutoModelForCausalLM.from_pretrained('data/thrml_spw/lfm3_1p7b/checkpoint',
        trust_remote_code=True, local_files_only=True, dtype=torch.bfloat16,
        attn_implementation='eager').cuda().eval()
    sys.modules[type(model.model.layers[0].conv).__module__].can_use_lfm3_conv = lambda *args: False
    targets = [(n,m) for n,m in model.named_modules() if m._parameters.get('weight') is not None]
    assert set(n for n,_ in targets) == set(aliases) and len(targets) == 287
    result, calls, handles = [], [], []

    def capture(name, module):
        def hook(mod, args, value):
            x = args[0]
            kind = ('embedding' if isinstance(module,torch.nn.Embedding) else
                'linear' if isinstance(module,torch.nn.Linear) else
                'conv' if isinstance(module,torch.nn.Conv1d) else 'elementwise')
            record = {'module':name, 'kind':kind, 'input_shape':list(x.shape),
                'output_shape':list(value.shape), 'weight_shape':list(module.weight.shape)}
            if kind == 'elementwise':
                program = load_program(records[aliases[name]]['program'])
                assert sha(records[aliases[name]]['program']) == records[aliases[name]]['program_sha256']
                normalized = (x.float()*torch.rsqrt(x.float().square().mean(-1,keepdim=True)+module.variance_epsilon)).to(x.dtype)
                raw = normalized.float().cpu().numpy()
                xx, layout, state = prepare(kind, raw, program)
                field = program['codes'].astype(np.float32)*np.float32(8/4095)-np.float32(4)
                probability = 1/(1+np.exp(-2*field.astype(np.float64)))
                width = int(program['group_size'])
                weight = (program['lower'].astype(np.float64)+program['span'].astype(np.float64)*probability.reshape(1,-1,width)).reshape(field.shape)
                # The independent canonical expression preserves the original
                # batch/sequence/head axes. No reserved IDs or model losses.
                expected = raw.astype(np.float64)*weight
                flat = (xx.astype(np.float64)*weight[None,None,:]).sum(1)
                actual = restore(flat,state)
                np.testing.assert_array_equal(actual,expected)
                assert actual.shape == tuple(value.shape)
                record.update(frontend_canonical_exact=True,
                    worker_input_shape=list(xx.shape),program_sha256=records[aliases[name]]['program_sha256'])
            calls.append(record)
        return hook

    try:
        for n,m in targets:
            handles.append(m.register_forward_hook(capture(n,m)))
        for batch,length in [(8,65),(4,257)]:
            calls.clear()
            ids=(torch.arange(batch*length,device='cuda').reshape(batch,length)*7919+17)%128000
            with torch.inference_mode():
                output=model(input_ids=ids,use_cache=False)
            assert Counter(v['module'] for v in calls) == Counter(aliases.keys())
            assert output.logits.shape == (batch,length,128000)
            result.append({'input_shape':[batch,length],'calls':list(calls),
                'output_shape':list(output.logits.shape)})
            print('complete synthetic forward',batch,length,flush=True)
            del output
    finally:
        for h in handles:h.remove()
    report={'status':'two_actual_model_forwards_all_287_modules_and_rank4_norm_frontend_verified',
        'sources_sha256':{n:sha(Path(__file__).with_name(n)) for n in
            ['probe_model_frontend_v3.py','holdout_frontend_v2.py','program_export.py']},
        'model_revision':'ac2586a34f1a1c03329dbdcdefabd8b6478ddaa7',
        'program_package_sha256':sha(root/'adaptation_pilot/mean/program_package.json'),
        'device':torch.cuda.get_device_name(0),'cases':result,
        'scope':'Actual model forwards and exact affine elementwise layout/restoration on synthetic integer IDs. No heldout tokens, token losses, quality selection, THRML sampling, or stochastic full-model proof.'}
    with (out/'results.json').open('x') as f:json.dump(report,f,indent=2)


if __name__=='__main__':main()
