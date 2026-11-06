"""Evaluate every frozen deterministic low-bit/mean control on consumed text."""
import json,time
from pathlib import Path
import numpy as np
import torch
from .sample_allocation_model import runner
from .fine_range_common import sha,atomic_json,atomic_npz
from .transformer_block_gpu import write_csv
from .fine_range_quantization_math import uniform,mean_weight,digest
OUT=Path('results/thrml_spw/fine_range_quantization_controls')
def main():
    p=json.loads((OUT/'protocol.json').read_text());assert sha(p['source_tokens'])==p['tokens_sha256'] and sha(OUT/'probability_codebook.npy')==p['probability_codebook_sha256']
    for name,h in p['source_sha256'].items():assert sha(name)==h
    assert not (OUT/'manifest.json').exists();tokens=np.load(p['source_tokens'])['test_tokens'];book=np.load(OUT/'probability_codebook.npy');r=runner(p);old=np.load(Path(p['source_cohort'])/'baseline_tokens.npz');rows=[];raw={};all_weights=[];start=time.monotonic();biases=sum(v.size for v in r.biases);assert sum(w.size for w in r.weights)==p['converted_weights'];outside=p['model_unique_parameters']-p['converted_weights']-biases
    for variant in p['variants']:
        weights=[];code_bits=0;range_bits=0;groups=0;g=variant.get('group',0)
        try:
            with torch.no_grad():
                for stage,(module,w) in enumerate(zip(r.modules,r.weights)):
                    if variant['kind']=='FP32':q=w;code_bits+=32*w.size
                    elif variant['kind']=='uniform':
                        q,meta=uniform(w,variant['bits'],g,variant['method']);code_bits+=meta['code_bits'];range_bits+=meta['range_metadata_bits'];groups+=meta['groups']
                    else:
                        c=p['control_artifacts'][str(g)][stage];assert sha(c['path'])==c['sha256'];a=np.load(c['path']);q=mean_weight(a,g,book);code_bits+=8*w.size;range_bits+=64*a['lower'].size;groups+=a['lower'].size
                    assert q.dtype==np.float32 and q.shape==w.shape and np.isfinite(q).all();module.weight.copy_(torch.from_numpy(q if module.__class__.__name__=='Conv1D' else q.T));weights.append({'stage':stage,'module':p['targets_in_forward_order'][stage],'shape':list(q.shape),'weight_sha256':digest(q)})
            value=r.evaluate(tokens)
            if variant['kind']=='FP32':np.testing.assert_array_equal(value['nll'],old['FP32_nll']);np.testing.assert_array_equal(value['predictions'],old['FP32_predictions'])
            raw[variant['name']+'_nll']=value['nll'];raw[variant['name']+'_predictions']=value['predictions'];flags=48 if g==16 else 0;table_bytes=book.nbytes if variant['kind']=='programmed_mean' else 0;bits=code_bits+range_bits+32*biases;assert bits%8==0;payload=bits//8
            row={'condition':variant['name'],'kind':variant['kind'],'method':variant.get('method',''),'bits':variant.get('bits',32 if variant['kind']=='FP32' else 8),'group':g,**value['scores'],'converted_weight_code_bits':code_bits,'range_metadata_bits':range_bits,'affine_bias_bits':32*biases,'group_count':groups,'code_range_bias_payload_bytes':payload,'configuration_bytes':flags,'global_lookup_bytes':table_bytes,'bits_per_converted_weight_including_affine_bias':bits/p['converted_weights'],'remaining_FP32_parameters_outside_payload':outside,'mixed_parameter_payload_bytes':payload+flags+table_bytes+4*outside,'stochastic_samples_generated':0};rows.append(row);all_weights.append({'condition':variant['name'],'weights':weights});print(json.dumps({'condition':variant['name'],**value['scores'],'elapsed_seconds':time.monotonic()-start}),flush=True)
        finally:r.restore()
    assert len(rows)==21;write_csv(OUT/'metrics.csv',rows);atomic_npz(OUT/'raw_tokens.npz',**raw);atomic_json(OUT/'weight_hashes.json',all_weights);manifest={'protocol_sha256':sha(OUT/'protocol.json'),'source_sha256':p['source_sha256'],'variants':21,'scores':rows,'raw_tokens_sha256':sha(OUT/'raw_tokens.npz'),'weight_hashes_sha256':sha(OUT/'weight_hashes.json'),'elapsed_seconds':time.monotonic()-start,'stochastic_samples_generated':0,'new_text_scored':False,'scope':p['scope']};atomic_json(OUT/'manifest.json',manifest);print(json.dumps({'complete':True,'variants':21,'new_native_samples':0}))
if __name__=='__main__':main()
