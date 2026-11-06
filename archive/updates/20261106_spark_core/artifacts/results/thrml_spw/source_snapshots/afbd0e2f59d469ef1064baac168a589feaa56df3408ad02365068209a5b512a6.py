"""Restore only missing comparison inputs; require original file hashes."""
from pathlib import Path
import os
import sys
import json
import hashlib
import concurrent.futures
import numpy as np


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(16*1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    os.chdir('/workspace');sys.path.insert(0,'/code')
    from research.thrml_spw.lfm3.adapted_word_conversion import convert_codes
    from research.thrml_spw.lfm3.program_export import load_program
    root=Path('results/thrml_spw/lfm3_1p7b')
    p=json.loads((root/'adapted_word_package/package.json').read_text())
    lookup={k:convert_codes(np.arange(4096,dtype=np.uint16),k) for k in [4,8]}
    def restore(rec):
        pending=[]
        for bits,item in rec['outputs'].items():
            path=Path(item['path'])
            if path.exists():assert sha(path)==item['sha256']
            else:pending.append((int(bits),item,path))
        if pending:
            source=Path(rec['original_program']);assert sha(source)==rec['original_program_sha256']
            a=load_program(source)
            for bits,item,path in pending:
                path.parent.mkdir(parents=True,exist_ok=True)
                temp=path.with_name(path.name+'.reconstructing')
                words=lookup[bits][a['codes']]
                assert hashlib.sha256(words.tobytes()).hexdigest()==item['words_sha256']
                with temp.open('xb') as f:
                    np.savez_compressed(f,words=words,lower=a['lower'],span=a['span'],shape=a['shape'],
                        group_size=a['group_size'],bits=np.asarray(bits,dtype=np.int64),
                        format_version=np.asarray(2,dtype=np.int64),sign_field_bound=np.asarray(4,dtype=np.int64))
                assert sha(temp)==item['sha256'],str(path)
                temp.replace(path)
        return len(pending)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        restored=sum(pool.map(restore,p['programs']))
    report={'status':'all_original_K4_K8_files_byte_exact_restored_or_verified','restored_files':restored,
        'total_files':572,'unique_parameters_per_width':1728801536,'scope':'Deterministic input reconstruction only. Every archive must match its original SHA256; no model sampling or altered scientific inputs.'}
    Path('word_reconstruction_v1.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
