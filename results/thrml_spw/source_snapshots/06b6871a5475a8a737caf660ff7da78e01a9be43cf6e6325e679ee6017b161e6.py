"""Export all quarter-model controls and replay their means on old development text.

No data from the ongoing conservative study is read or scored by this artifact test.
"""
import csv,json
from pathlib import Path
import numpy as np
from .packed_controls import save_controls,load_controls,expected_weights
from .verify_transformer_control import sha


def main():
    root=Path('results/thrml_spw');source=root/'transformer_quarter';out=root/'packed_quarter_controls';out.mkdir(exist_ok=True)
    weights=np.load(source/'weights.npz');rows=[]
    for i in range(12):
        rows.append({'stage':i,**save_controls(out/f'stage{i}.npz',weights[f'weight_{i}'],weights[f'bias_{i}'])})
    with (out/'storage.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    # Independent vector-level field/code/mode checks over all21.23M weights.
    for i,row in enumerate(rows):
        loaded=load_controls(row['path']);w=weights[f'weight_{i}'];grouped=w.reshape(-1,128,w.shape[1])
        lower=grouped.min(1);span=grouped.max(1)-lower
        np.testing.assert_array_equal(loaded['lower'],lower);np.testing.assert_array_equal(loaded['span'],span)
        lo=np.repeat(lower,128,0);sp=np.repeat(span,128,0);p=np.zeros_like(w);np.divide(w-lo,sp,out=p,where=sp!=0);p=np.clip(p,0,1).astype(np.float64).ravel()
        free=(p>0)&(p<1);original_field=((np.log(p[free])-np.log1p(-p[free]))/2).astype(np.float32).astype(np.float64)
        code=np.rint((np.maximum(-4,np.minimum(4,original_field))+4)*255/8).astype(np.uint8)
        requested=(-4+code.astype(np.float64)*8/255).astype(np.float32)
        np.testing.assert_array_equal(loaded['codes'],code);np.testing.assert_array_equal(loaded['fields'],requested);np.testing.assert_array_equal(loaded['free_indices'],np.flatnonzero(free))
        expected_mode=np.zeros(w.size,dtype=np.uint8);expected_mode[p==0]=1;expected_mode[p==1]=2;np.testing.assert_array_equal(loaded['modes'],expected_mode)
    import torch
    from huggingface_hub.constants import HF_HUB_CACHE
    from transformers import AutoModelForCausalLM
    from .sequential_transformer import SequentialRunner
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    protocol=json.loads((source/'protocol.json').read_text());snapshot=Path(HF_HUB_CACHE)/f"models--distilbert--distilgpt2/snapshots/{protocol['revision']}"
    runner=SequentialRunner(AutoModelForCausalLM.from_pretrained(snapshot,local_files_only=True,attn_implementation='eager').float().eval(),protocol['targets_in_forward_order'])
    with torch.no_grad():
        for i,module in enumerate(runner.modules):module.weight.copy_(torch.from_numpy(expected_weights(load_controls(out/f'stage{i}.npz')).astype(np.float32)))
    old=np.load('data/thrml_spw/transformer_projection/export.npz');replay=runner.evaluate(old['validation_tokens']);raw=np.load(source/'baseline_tokens.npz')
    # CPU thread partitions can alter the final float32 reduction at tiny scale.
    np.testing.assert_allclose(replay['nll'],raw['validation_programmed_mean8_nll'],rtol=0,atol=2e-5)
    np.testing.assert_array_equal(replay['predictions'],raw['validation_programmed_mean8_predictions'])
    np.savez_compressed(out/'mean_replay.npz',nll=replay['nll'],predictions=replay['predictions'])
    payload=sum(r['numeric_payload_bytes'] for r in rows);actual=sum(r['archive_bytes'] for r in rows);fp=sum(r['nominal_FP32_weight_bias_bytes'] for r in rows)
    result={'passed':True,'matrices':12,'weight_count':sum(r['weights'] for r in rows),'numeric_payload_bytes':payload,'actual_artifact_bytes':actual,
        'nominal_original_FP32_weight_bias_bytes':fp,'nominal_FP32_over_actual_artifact_ratio':fp/actual,
        'numeric_payload_bits_per_weight':payload*8/sum(r['weights'] for r in rows),'actual_artifact_bits_per_weight':actual*8/sum(r['weights'] for r in rows),
        'prior_conventional_affine8_nominal_bits_per_weight':8.34375,'prior_conventional_affine4_nominal_bits_per_weight':4.3125,
        'mean_replay':replay['scores'],'max_replay_loss_difference':float(np.max(np.abs(replay['nll']-raw['validation_programmed_mean8_nll']))),
        'source_master_sha256':sha(source/'weights.npz'),'artifact_sha256':{r['path']:sha(r['path']) for r in rows},
        'source_sha256':sha(__file__),'encoder_source_sha256':sha(Path(__file__).with_name('packed_controls.py')),
        'scope':'Actual compact parameter artifacts, vector equality over every code/mode/field and deterministic mean replay on previously scored development text. CPU/CUDA tiny native same-path tests are separate. Artifact payload stores no original target matrices; sampler still expands float32 fields and a large program. Not full stochastic-model packed execution, runtime compression, silicon area or energy proof.'}
    (out/'REPORT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
