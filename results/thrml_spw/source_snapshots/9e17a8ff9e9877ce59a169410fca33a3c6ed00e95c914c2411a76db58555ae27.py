"""Freeze real-model synthetic replacement/replay fixtures, without test data."""
from pathlib import Path
import hashlib,json
import numpy as np
from .r1_io_v1 import BASE,sha


def main():
    output=BASE/'r1_model_preflight_v2';output.mkdir(exist_ok=False)
    packages=json.loads((BASE/'additional_parameter_packages_v1_results.json').read_text())
    assets=json.loads((BASE/'additional_assets_v1_manifest.json').read_text())
    models={};cases={};rng=np.random.default_rng(202611963)
    for index,package in enumerate(packages['packages']):
        key='gpt2' if index==0 else 'distilbert';length=65 if key=='gpt2' else 256
        ids=rng.integers(0,1000,(2,length),dtype=np.int64);arrays={'input_ids':ids}
        if key=='distilbert':
            mask=np.ones_like(ids);mask[1,-7:]=0;ids[1,-7:]=0;arrays['attention_mask']=mask
        path=output/(key+'_synthetic.npz')
        with path.open('xb') as f:np.savez_compressed(f,**arrays)
        models[key]={'model_type':key,'package':package['path'],'package_sha256':package['sha256'],
            'checkpoint':assets['checkpoint_locations'][package['model']],
            'data':{'kind':'synthetic','arrays':str(path)}}
        for arm in ['original','mean1','mean4','mean8','int2','int4','int8']:
            cases[key+'__'+arm]={'model_key':key,'mode':'control','arm':arm}
        cases[key+'__nativeT1']={'model_key':key,'mode':'native','bits':1,'samples':1,'budget':1,'seed':202611960+index}
    names=['r1_io_v1.py','r1_worker_v1.py','r1_model_v1.py','r1_case_v1.py','verify_r1_case_v1.py',
        'prepare_r1_preflight_v2.py','parameter_program_v1.py','parameter_counts_v1.py','parameter_counts_v2.py']
    sources=[Path(__file__).with_name(n) for n in names]
    sources+=[Path('research/thrml_spw')/n for n in ['native.py','controlled_positive_native.py','gpu_environment.py']]
    sources+=[Path('research/thrml_spw/lfm3/counted_bank.py'),Path('research/thrml_spw/lfm3/stage_checkpoint_v1.py')]
    sources=[x.relative_to(Path.cwd()) if x.is_absolute() else x for x in sources]
    inputs=[BASE/n for n in ['additional_parameter_packages_v2_audit.json','large_tile_h100_v1_audit.json','parameter_counts_h100_v1_audit.json']]
    inputs += [Path(s['package']) for s in models.values()]
    # Local mirrors preserve exact remote package bytes, not the original arrays.
    for s in models.values():
        target=Path(s['package']);mirror=BASE/'remote_metadata'/target.relative_to(BASE)
        assert sha(mirror)==s['package_sha256']
        inputs.remove(target);inputs.append(mirror)
    inputs += [Path(s['data']['arrays']) for s in models.values()]
    p={'status':'actual_model_synthetic_preflight_no_task_scoring','output':str(output),'models':models,'cases':cases,
        'sources_sha256':{str(s):sha(s) for s in sources},'inputs_sha256':{str(s):sha(s) for s in inputs},
        'scope':'Both actual pinned models, all seven deterministic controls and a fullparameter officialTHRMLT1draw. Only synthetic integer inputs; no evaluationtokens or taskquality.'}
    assert all(not Path(x).is_absolute() for x in [*p['sources_sha256'],*p['inputs_sha256']])
    for s in sources:
        target=Path('results/thrml_spw/source_snapshots',sha(s)+'.py')
        if not target.exists():target.write_bytes(s.read_bytes())
    with (output/'protocol.json').open('x') as f:json.dump(p,f,indent=2);f.write('\n')
    print(json.dumps({'status':'synthetic_preflight_frozen','cases':len(cases),'protocol_sha256':sha(output/'protocol.json')}))

if __name__=='__main__':main()
