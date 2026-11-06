"""Run the frozen native study and its exact64-window reporting finalizer.

Retains the original driver and its recorded source hash. Only the known obsolete
128-window reporting guard is handled, after verifying all15 native runs finished.
Any other exception or incomplete native phase is propagated unchanged.
"""
import json,traceback
from pathlib import Path
from .verify_transformer_control import sha


def main():
    from .calibrated_transformer import main as run
    try:run()
    except ValueError as error:
        if str(error)!='Expected five seeds and128 windows of64 target positions':raise
        out=Path('results/thrml_spw/calibrated_transformer');p=json.loads((out/'protocol.json').read_text())
        if p['windows']!=64:raise
        for condition in p['conditions']:
            for seed in range(5):
                path=out/'runs'/f'{condition}_seed{seed}'/'run.json'
                if not path.exists() or json.loads(path.read_text())['status']!='complete':raise
        (out/'reporting_guard_recovery.json').write_text(json.dumps({'recovered_known_reporting_guard':True,'native_cases_complete':15,'original_error':str(error),'traceback':traceback.format_exc(),'wrapper_sha256':sha(__file__),'scope':'No scientific parameters or raw native cases changed. Only original hardcoded128-window final reporting rejected this fixed64-window study.'},indent=2)+'\n')
        from .finalize_calibrated_transformer import main as finish
        finish()


if __name__=='__main__':main()
