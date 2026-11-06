"""Measure metric-reduction drift where H100 control arrays were bit-identical."""
from pathlib import Path
import hashlib
import json
import numpy as np


def main():
    base=Path('results/thrml_spw/broad_verification_20261105')
    root=base/'h100_mirror/results/thrml_spw/lfm3_1p7b/modal_h100_word_v2/C8192'
    rows=[];inputs={}
    for bits in [4,8]:
        reference=None;maximum=0.;changed=0
        for seed in range(202610600,202610605):
            folder=root/f'K{bits}/mean/{seed}'
            ap,rp=folder/'h100_independent_audit_v3.json',folder/'results.json'
            a=json.loads(ap.read_text());r=json.loads(rp.read_text())
            assert a['all_H100_BF16_controls_bitwise_equal_to_H100_master']
            assert hashlib.sha256(rp.read_bytes()).hexdigest()==a['results_sha256']
            inputs[str(rp)]=a['results_sha256'];inputs[str(ap)]=hashlib.sha256(ap.read_bytes()).hexdigest()
            values=np.asarray([m['token_NLL'] for m in r['metrics'] if m['stage']=='same_fit_BF16'])
            if reference is None:reference=values
            maximum=max(maximum,float(np.abs(values-reference).max()))
            changed+=int(np.count_nonzero(values!=reference))
        rows.append({'word_bits':bits,'maximum_absolute_per_token_NLL_reduction_difference':maximum,
            'entries_different_from_first_seed_control':changed})
    report={'status':'identical_control_tensors_have_nonidentical_FP64_metric_reductions',
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'inputs_sha256':inputs,'observations':rows,
        'engineering_requirement':'Recovery implementation must preserve per-seed losses and allow at most1e-12absolute numerical variation for deterministic FP64 control-loss validation. Keep own-H100 raw control tensor equality, original2percentqualitymargin and paired statistical definitions. Do not round or discard losses.',
        'scope':'Diagnostic on already completed fixed-validation comparisons, not new holdout outcomes or a change in task-quality criteria.'}
    with (base/'control_reduction_diagnostic_v1.json').open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps(rows))


if __name__=='__main__':main()
