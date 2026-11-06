"""Single source of transport paths; scientific v4 case code remains frozen."""
from pathlib import Path
import hashlib,json
APP_NAME='pbit1-broad-heldout-v5'
LOG_DIRECTORY='cloud_holdout_logs_v5'
CORE_PROTOCOL='results/thrml_spw/lfm3_1p7b/holdout_native_v4_protocol.json'
CORE_ASSETS='holdout_assets_v4.json'
TRANSPORT='results/thrml_spw/broad_verification_20261105/holdout_transport_v5.json'
CASE_MODULES=('holdout_native_case_v4','verify_holdout_native_case_v4')
STATISTICS_MODULES=('holdout_native_statistics_v4','verify_holdout_native_statistics_v4')


def validate_transport(root, digest):
    root=Path(root)
    path=root/TRANSPORT
    assert hashlib.sha256(path.read_bytes()).hexdigest()==digest
    p=json.loads(path.read_text())
    assert p['status']=='transport_only_amendment_frozen_before_core_v4_inference'
    assert p['app']==APP_NAME and p['logs']==LOG_DIRECTORY and p['core_protocol']==CORE_PROTOCOL
    assert p['core_assets']==CORE_ASSETS
    assert p['case_modules']==list(CASE_MODULES) and p['statistics_modules']==list(STATISTICS_MODULES)
    assert hashlib.sha256((root/CORE_PROTOCOL).read_bytes()).hexdigest()==p['core_protocol_sha256']
    for name,digest in p['sources_sha256'].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
    return p
