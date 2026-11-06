"""Resume an incomplete diagnostic or verify a completed immutable cached study."""
from pathlib import Path
import json
from .verify_transformer_control import sha
OUT=Path('results/thrml_spw/transformer_calibration_floor')


def main():
    manifest=OUT/'manifest.json'
    if not manifest.exists():
        from .transformer_calibration_floor import main as run
        run();return
    m=json.loads(manifest.read_text());p=json.loads((OUT/'protocol.json').read_text());assert m['config']==p
    for n,h in p['source_sha256'].items():assert sha(n)==h
    assert sha(Path(p['source_study'])/'manifest.json')==p['source_manifest_sha256'] and sha(p['tokens_path'])==p['tokens_sha256']
    for seed in range(5):
        r=json.loads((OUT/'runs'/f'oracle8_seed{seed}'/'run.json').read_text());assert r['status']=='complete' and len(r['stages'])==12
        for s in r['stages']:
            meta=json.loads(Path(s['metadata']).read_text());assert sha(s['inputs'])==meta['signature']['inputs_sha256'] and sha(s['output'])==meta['output_sha256']
    for path in (OUT/'mean_runs').glob('*.json'):
        meta=json.loads(path.read_text());assert sha(path.with_suffix('.npz'))==meta['archive_sha256']
    assert len(list((OUT/'mean_runs').glob('*.json')))==16
    result={'passed':True,'completed_native_runs_verified':5,'native_operator_artifacts_verified':60,'mean_runs_verified':16,'manifest_sha256':sha(manifest),'manifest_rewritten':False,'new_native_samples':0,'source_sha256':sha(__file__),
        'scope':'Verify completed signature/hash cache without calling the elapsed-time-writing original driver, resampling or changing immutable audited results. Incomplete studies resume through the original frozen driver.'}
    (OUT/'cached_reproduction_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
