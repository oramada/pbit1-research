"""One frozen cloud spatial-control case, with independent audit and no retry."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import hashlib
import numpy as np


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--budget',type=int,required=True)
    parser.add_argument('--bits',type=int,required=True)
    parser.add_argument('--seed',type=int,required=True)
    args=parser.parse_args()
    extra=['--arm','mean','--budget',str(args.budget),'--word-bits',str(args.bits),'--seed',str(args.seed)]
    for module in ['modal_h100_word_v2','verify_modal_h100_word_v2']:
        subprocess.run([sys.executable,'-m','research.thrml_spw.lfm3.'+module,*extra],check=True)
        if module=='modal_h100_word_v2' and (args.budget,args.bits,args.seed)==(1024,8,202610350):
            gold=json.loads(Path('h100_parity_reference_v1.json').read_text())
            out=Path('results/thrml_spw/lfm3_1p7b/modal_h100_word_v2/C1024/K8/mean/202610350')
            differences=[]
            for batch in range(4):
                with np.load(out/f'batch{batch}.npz') as arrays:
                    for stage in ['BF16','same_fit_BF16','nominal_controls','native']:
                        a=arrays[stage]
                        if hashlib.sha256(a.tobytes()).hexdigest()!=gold['array_sha256'][f'{batch}/{stage}']:
                            differences.append(f'{batch}/{stage}')
            (out/'cross_backend_parity.json').write_text(json.dumps({'status':'exact_all_stage_replay' if not differences else 'cross_backend_difference_preserved','different_arrays':differences,'reference_results_sha256':gold['reference_results_sha256']},indent=2)+'\n')
            assert not differences,'H100 whole-model replay differs; do not mix local/cloud cohort'
    print(json.dumps({'status':'cloud_case_and_independent_audit_complete','budget':args.budget,'bits':args.bits,'seed':args.seed}),flush=True)


if __name__=='__main__':main()
