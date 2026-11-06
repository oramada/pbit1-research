"""Observed-update and storage ledger, including all selection/audit overhead."""
import json
from pathlib import Path
import numpy as np
from .transformer_block_gpu import sha,save_json
OUT=Path('results/thrml_spw/sample_allocation')
def main():
    p=json.loads((OUT/'protocol.json').read_text());a=json.loads((OUT/'allocation.json').read_text());val=json.loads((OUT/'validation_manifest.json').read_text());test=json.loads((OUT/'test_manifest.json').read_text());va=json.loads((OUT/'validation_native_audit.json').read_text());ta=json.loads((OUT/'test_native_audit.json').read_text());assert va['passed'] and ta['passed'];n=42467328;perbank=n*1024;assert a['weighted_updates_per_bank']==perbank;assert val['native_observed_updates']==5*4*256*n and test['native_observed_updates']==10*4*1024*n
    failed=4*256*(5*(p['control_artifacts'][0]['weights']+p['control_artifacts'][1]['weights'])+p['control_artifacts'][2]['weights']);assert failed==14495514624
    packed=[];rawbytes=0
    for condition in ['uniform','allocated']:
        total=0
        for seed in range(5):
            folder=OUT/'test/runs'/f'{condition}_seed{seed}';r=json.loads((folder/'run.json').read_text());total+=r['free_node_updates']
            for s in r['stages']:
                path=Path(s['output']);meta=json.loads(Path(s['metadata']).read_text());assert sha(path)==meta['output_sha256'];rawbytes+=int(np.prod(meta['output_shape']))*4;packed.append(path.stat().st_size)
        assert total==5*4*perbank
    ledger={'validation_native_operators':120,'test_native_operators':240,'validation_selection_observed_updates':val['native_observed_updates'],'test_production_observed_updates':test['native_observed_updates'],'validation_public_output_and_frequency_replay_updates':va['observed_node_updates_replayed'],'test_public_output_and_frequency_replay_updates':ta['observed_node_updates_replayed'],'failed_combined_observer_replay_updates':failed,'total_executed_native_observed_updates':val['native_observed_updates']+test['native_observed_updates']+va['observed_node_updates_replayed']+ta['observed_node_updates_replayed']+failed,
        'test_updates_per_condition_per_seed':4*perbank,'test_updates_per_condition':5*4*perbank,'converted_free_nodes':n,'converted_fraction':n/81912576,'clamped_weight_nodes':0,'weighted_mean_samples_each_condition':1024.,'uniform_sum_layer_samples':24576,'allocated_sum_layer_samples':sum(a['samples']),'uniform_max_layer_samples':1024,'allocated_max_layer_samples':max(a['samples']),'uniform_serial_sampling_rounds_per_test_seed':4*24576,'allocated_serial_sampling_rounds_per_test_seed':4*sum(a['samples']),
        'control_range_bias_payload_bytes':sum(c['payload_bytes'] for c in p['control_artifacts']),'allocation_schedule_uint16_bytes':48,'uniform_sample_scalar_uint16_bytes':2,'lossless_packed_test_output_files':len(packed),'lossless_packed_test_output_bytes':sum(packed),'same_arrays_unpacked_float32_bytes':rawbytes,'packed_over_raw_array_ratio':sum(packed)/rawbytes,'source_sha256':sha(__file__),
        'scope':'All counted updates are simulator node redraws, not necessarily bit changes or physical switching. Includes selection, failed combined-observer check, and separate output/frequency verification passes. Equal test weighted updates/device count does not establish equal energy, latency or total system memory. SumT is serialized layer sampling rounds; maxT also changes. Packing is software evidence storage only and preserves every F32 bit; model encoding/control payload unchanged.'}
    save_json(OUT/'resource_ledger.json',ledger);print(json.dumps(ledger,indent=2))
if __name__=='__main__':main()
