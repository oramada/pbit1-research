"""Logical LFM3 representation payload accounting, excluding physical claims."""
import hashlib
import json
from pathlib import Path


def main():
    root=Path('results/thrml_spw/lfm3_1p7b')
    meta=json.loads((root/'meta_inventory.json').read_text())
    groups=sum(p['shape'][0] if len(p['shape'])>1 else 1 for p in meta['parameters'])
    n=sum(p['numel'] for p in meta['parameters'])
    result={'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'input_sha256':hashlib.sha256((root/'meta_inventory.json').read_bytes()).hexdigest(),
        'unique_weights':n,'affine_groups':groups,'logical_pbits_if_fully_resident':n,
        'packed_binary_state_bytes':(n+7)//8,'eight_bit_programming_commands_bytes':n,
        'BF16_reference_weight_bytes':2*n,'two_FP32_metadata_values_per_group_bytes':8*groups,
        'two_FP64_metadata_values_per_group_bytes':16*groups,
        'packed_state_plus_8bit_commands_plus_FP32_metadata_bytes':(n+7)//8+n+8*groups,
        'simulator_unpacked_bool_state_bytes_if_fully_resident':n,
        'simulator_FP32_unary_field_bytes_if_fully_resident':4*n,
        'scope':'Bookkeeping for same row-wise affine policy, global per-vector norm groups, tied embedding/head counted once. Packed state is theoretical logical storage; actual simulator bool is unpacked. Does not include original checkpoint retained by software, node objects, framework indices, activations, KV/convolution caches, accumulators, compiler workspace, readout, reprogramming or energy. No measured total memory or hardware advantage.'}
    (root/'resource_inventory.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
