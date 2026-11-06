"""Independent checkpoint-shape and packed resource ledger verification."""
from pathlib import Path
import json,hashlib
import numpy as np
from huggingface_hub.constants import HF_HUB_CACHE
OUT=Path('results/thrml_spw/positional_student')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
    p=json.loads((OUT/'protocol.json').read_text());ledger=json.loads((OUT/'resource_ledger.json').read_text());config=Path(HF_HUB_CACHE)/f"models--distilbert--distilgpt2/snapshots/{p['revision']}/config.json";cfg=json.loads(config.read_text());d,L,V,P=[cfg[k] for k in ['n_embd','n_layer','vocab_size','n_positions']];assert cfg.get('n_inner') is None and cfg.get('tie_word_embeddings',True);total=V*d+P*d+L*(12*d*d+13*d)+2*d;assert total==p['model_unique_parameters']==81912576;converted=0;biases=0;payloads={1:0,4:0}
    for entry in p['matrices']:
        n=int(np.prod(entry['shape']));converted+=n;biases+=entry['shape'][1]
        for k,key in [(1,'single_controls'),(4,'positional_controls')]:
            c=entry[key];assert sha(c['path'])==c['sha256'];a=np.load(c['path']);assert a['codes'].shape==tuple(entry['shape']) and a['codes'].dtype==np.uint8 and a['bias'].size==entry['shape'][1];assert a['lower'].shape==a['span'].shape==(entry['shape'][0]//16,entry['shape'][1]);assert a['lower'].dtype==a['span'].dtype==a['bias'].dtype==np.float32
            if k==4:assert a['codes'].max()<=15
            payloads[k]+=n*(8 if k==1 else 4)+8*(a['lower'].nbytes+a['span'].nbytes+a['bias'].nbytes)
    assert converted==42467328 and biases==41472 and total-converted-biases==39403776;derived=[]
    for cfg,row in zip(p['conditions'],ledger['rows']):
        k,t=cfg['K'],cfg['T'];payload=payloads[k]//8;outside=4*(total-converted-biases);expected={'K':k,'T':t,'converted_weights':converted,'converted_weight_fraction':converted/total,'total_converted_dynamic_nodes_if_all_resident':k*converted,'max_layer_dynamic_nodes':k*max(e['weights'] for e in p['matrices']),'node_updates_per_weight_per_bank':k*t,'node_updates_per_32context_forward':4*k*t*converted,'node_updates_all5seeds':20*k*t*converted,'serialized_projection_sample_rounds_per_bank':24*t,'serialized_projection_sample_rounds_all4banks':96*t,'control_range_bias_payload_bits_per_weight':payloads[k]/converted,'control_range_bias_payload_bytes':payload,'group_configuration_bytes':48,'remaining_deterministic_FP32_parameter_bytes':outside,'mixed_serialized_parameter_bytes_excluding_dynamic_state':payload+48+outside,'dynamic_state_bits_if_all_weights_resident':k*converted}
        for name,value in expected.items():assert row[name]==value,(name,row[name],value)
        derived.append({'condition':cfg['name'],'mixed_parameter_payload_bytes':payload+48+outside,'packed_dynamic_state_bytes':k*converted//8,'mixed_parameter_plus_packed_state_bytes':payload+48+outside+k*converted//8,'converted_parameter_plus_state_bits_per_weight':(payloads[k]+k*converted)/converted})
    assert sum(v['node_updates_all5seeds'] for v in ledger['rows'])==ledger['production_node_updates']==434865438720;assert ledger['independent_native_replay_updates']==2*ledger['production_node_updates'];r={'passed':True,'independent_unique_parameter_count_from_pinned_config':total,'converted_biases_counted_once':biases,'all_resource_fields_verified':True,'source_config_sha256':sha(config),'resource_ledger_sha256':sha(OUT/'resource_ledger.json'),'producer_protocol_sha256':sha(OUT/'protocol.json'),'additional_logical_storage_comparisons':derived,'source_sha256':sha(__file__),'scope':'Nominal serialized bits plus packed dynamic states, not measured RAM, device area, power or latency. Physical stochastic elements are not interchangeable with ordinary digital storage cells.'};(OUT/'resource_audit.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
if __name__=='__main__':main()
