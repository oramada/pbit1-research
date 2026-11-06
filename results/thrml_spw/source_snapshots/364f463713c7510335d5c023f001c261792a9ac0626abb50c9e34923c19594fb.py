"""Static fixed-command/range compatibility and bounded payload review."""
import hashlib
import json
from pathlib import Path

import numpy as np

from results.thrml_spw.fashion_cnn_native.probability_native.adapter import encoding

ROOT = Path('results/thrml_spw/fashion_cnn_native')
OUT = ROOT/'probability_native_evidence'


def main():
    OUT.mkdir(exist_ok=True)
    assert not (OUT/'preparation_check.json').exists()
    original = json.loads((ROOT/'training/manifest.json').read_text())
    fine = json.loads((ROOT/'probability_finetune_evidence/manifest.json').read_text())
    assert fine['complete'] and len(fine['rows']) == 10
    checked = 0
    rows = []
    for row in fine['rows']:
        checkpoint = np.load(row['path'])
        source = np.load(original['rows'][row['model']]['path'])
        enc = encoding(checkpoint)
        for layer, prefix in enumerate(['conv1', 'conv2', 'head']):
            w = source[prefix+'_weight'].astype(float).reshape(len(source[prefix+'_weight']), -1)
            c, r = (w.max(1)+w.min(1))/2, (w.max(1)-w.min(1))/2
            e = enc[layer]
            np.testing.assert_array_equal(e['center'], c)
            np.testing.assert_array_equal(e['radius'], r)
            np.testing.assert_array_equal(e['bias'], source[prefix+'_bias'])
            code = np.round((checkpoint[prefix+'_latent_fields']+4)*255/8).astype(np.uint8)
            np.testing.assert_array_equal(e['commands'], code)
            expected = np.asarray([[-4+float(k)*8/255 for k in values] for values in code], np.float32)
            np.testing.assert_array_equal(e['fields'], expected.T)
            assert e['fields'].shape == (w.shape[1], w.shape[0])
            assert np.isfinite(e['fields']).all() and np.max(abs(e['fields'])) <= 4
            checked += code.size
        np.testing.assert_array_equal(checkpoint['activation_span'], source['activation_span'].astype(np.float32).astype(float))
        np.testing.assert_array_equal(checkpoint['activation_lower'], np.zeros(3))
        rows.append({'model': row['model'], 'method': row['method'], 'weight_commands_checked': 2664})
    assert checked == 26640
    counter_entries_per_bank = (2*8*14*14*8+8)+(2*8*7*7*16+16)+(8*10+10)
    assert counter_entries_per_bank == 37746
    max_counter = 255*144*1024
    assert max_counter == 37601280 and max_counter < np.iinfo(np.int32).max
    counter_bytes = 2*counter_entries_per_bank*4
    code_bytes = 2*(8*14*14*9+8*7*7*72+8*144)
    mask_bytes = 2*(8*14*14*9+8*7*7*72)
    output_bytes = 2*(8*8*7*7+8*16*3*3+8*10)*8
    assert counter_bytes+code_bytes+mask_bytes+output_bytes == 543504
    result = {'passed': True, 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'rows': rows, 'weight_commands_checked': checked, 'counter_entries_per_bank': counter_entries_per_bank,
              'max_counter_value_T1024': max_counter, 'raw_case_array_bytes': 543504,
              'raw_30case_array_bytes': 16305120, 'array_and_support_storage_bound_bytes': 20000000,
              'scope': 'All30layer exports/26640commands and fixed-range/bias/activation metadata checked before source freeze; array payload arithmetic, not native execution or physical memory cost.'}
    (OUT/'preparation_check.json').write_text(json.dumps(result, indent=2)+'\n')
    print('All30learned layer encodings/26640commands checked; 30-case raw arrays16.31MB')


if __name__ == '__main__':
    main()
