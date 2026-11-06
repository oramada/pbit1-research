"""Correct wrapper log namespace without changing frozen scientific source."""
from pathlib import Path
import hashlib,json,ast
from datetime import datetime,timezone
from .layout import APP_NAME,LOG_DIRECTORY,CORE_PROTOCOL,CORE_ASSETS,TRANSPORT,CASE_MODULES,STATISTICS_MODULES,validate_transport

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    root=Path('cloud/broad_validation_v5')
    # Validate actual launch constants and every wrapper reference, including the
    # bad log path that the prior terminal guard caught before core inference.
    source=(root/'app.py').read_text();tree=ast.parse(source)
    assert 'cloud_holdout_logs_v3' not in source and 'cloud_holdout_logs_v4' not in source
    assert source.count('LOG_DIRECTORY')==3
    assert 'for module in CASE_MODULES:' in source and 'for module in STATISTICS_MODULES:' in source
    assert 'validate_transport' in source and 'transport_sha256' in source
    assert LOG_DIRECTORY=='cloud_holdout_logs_v5' and APP_NAME.endswith('-v5')
    p={'status':'transport_only_amendment_frozen_before_core_v4_inference',
       'recorded_utc':datetime.now(timezone.utc).isoformat(),'app':APP_NAME,'logs':LOG_DIRECTORY,
       'core_protocol':CORE_PROTOCOL,'core_protocol_sha256':sha(CORE_PROTOCOL),'core_assets':CORE_ASSETS,
       'case_modules':list(CASE_MODULES),'statistics_modules':list(STATISTICS_MODULES),
       'prior_terminal_call_id':'fc-01M6PSMSANJSMMD1G3N929HZN0',
       'prior_terminal_reason':'Reused v3logdirectory correctly rejected by terminal failure guard before starting frozen v4core. No native sampling or v4case output was produced.',
       'engineering_change':'Only deployed app/claim/log/observer namespace corrected. All corev4sources, protocol, rawdata, seeds, margin and stage checkpoint behavior remain byte-identical. No extra MC trial or score selection.',
       'sources_sha256':{str(f):sha(f) for f in root.glob('*.py')}}
    with Path(TRANSPORT).open('x') as f:json.dump(p,f,indent=2);f.write('\n')
    digest=sha(TRANSPORT);validate_transport('.',digest)
    for f in root.glob('*.py'):
        target=Path('results/thrml_spw/source_snapshots',sha(f)+'.py')
        if not target.exists():target.write_bytes(f.read_bytes())
    print(json.dumps({'status':'transport_v5_paths_and_source_binding_validated','transport_sha256':digest,'core_protocol_sha256':p['core_protocol_sha256']}))

if __name__=='__main__':main()
