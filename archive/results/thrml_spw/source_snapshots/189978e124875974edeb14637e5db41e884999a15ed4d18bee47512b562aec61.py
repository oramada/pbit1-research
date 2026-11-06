"""Freeze selected quarter-model confirmation settings before scoring new text."""
import hashlib,json
from pathlib import Path


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    root=Path('results/thrml_spw');prep=root/'quarter_confirmation_preparation'
    development=root/'transformer_quarter';selection_path=development/'selection.json';selection=json.loads(selection_path.read_text());assert selection['gate_passed']
    source=json.loads((development/'protocol.json').read_text());reserve=json.loads((prep/'protocol.json').read_text());audit=json.loads((prep/'reservation_audit.json').read_text());assert audit['passed']
    protocol={'model':source['model'],'revision':source['revision'],'dataset_revision':source['dataset_revision'],
        'targets_in_forward_order':source['targets_in_forward_order'],'converted_matrix_weights':source['converted_matrix_weights'],'converted_fraction':source['converted_fraction'],
        'selected_samples':selection['selected_samples'],'field_bits':8,'field_bound':4.,'input_block_size':128,
        'contexts_per_weight_bank_batch':8,'sampling_seeds':list(range(5)),'confirmation_domain':990730,
        'input_data_path':'data/thrml_spw/quarter_confirmation/tokens.npz','input_tokens_key':'tokens','input_export_sha256':reserve['tokens_sha256'],
        'selection_sha256':sha(selection_path),'development_protocol_sha256':sha(development/'protocol.json'),
        'reservation_protocol_sha256':sha(prep/'protocol.json'),'reservation_audit_sha256':sha(prep/'reservation_audit.json'),
        'conditions':['FP32','affine4_block128','affine8_block128','programmed_mean8','native_finite8'],
        'decision':'Report fixed native point-mean CE increase<=log(1.05) and token-accuracy loss<=0.01 versus same-text FP32, with every outcome retained. No sample/precision/grouping/model selection on confirmation scores.',
        'uncertainty':'Normal conditional MC intervals and10000 crossed paired seed/bank bootstrap resamples, five seeds and sixteen banks of eight contexts; bootstrap seed20261108. Sensitivity under exchangeable clusters, not distribution-free corpus coverage.',
        'confirmation_used_for_tuning':False,'scope':'Frozen quarter-model configuration on new positions in the same WikiText-2 test corpus. Affine normalized p-bit encoding; not full-model or physical-device proof.',
        'source_sha256':sha(__file__)}
    path=prep/'frozen_evaluation.json'
    if path.exists():assert json.loads(path.read_text())==protocol,'Do not mutate frozen confirmation'
    else:path.write_text(json.dumps(protocol,indent=2)+'\n')
    print(json.dumps({'selected_samples':protocol['selected_samples'],'converted_fraction':protocol['converted_fraction'],'confirmation_scores_computed':False,'frozen_protocol_sha256':sha(path)}))

if __name__=='__main__':main()
