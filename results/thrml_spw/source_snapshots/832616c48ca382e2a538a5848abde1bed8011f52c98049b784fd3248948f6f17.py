"""Verify the benchmark auditor rejects corrupted copies, never touching originals."""
from pathlib import Path
import json,tempfile
import numpy as np
from .prepare_v1 import OUT,sha
from .audit_benchmark_v1 import inspect_case

def main():
    root=OUT/'benchmark_v1';p=json.loads((root/'protocol.json').read_text());case=next(c for c in p['cases']if c['name']=='selected_seed2026110801')
    layout=json.loads(Path(p['data']['layout']).read_text())
    with np.load(p['data']['tokens'])as d:tokens=d['tokens'].copy();offsets=d['offsets'].copy()
    baseline=inspect_case(root,case,p,tokens,offsets,layout);results=[]
    for fixture in ['input_identity','summary_metric','missing_last_window','sample_budget','layout_choice']:
        with tempfile.TemporaryDirectory(prefix='pbit_audit_fixture_')as tmp:
            fake=Path(tmp);folder=fake/case['name'];folder.mkdir()
            for path in (root/case['name']).iterdir():
                if path.is_file():(folder/path.name).symlink_to(path.resolve())
            if fixture=='missing_last_window':(folder/'window0585.json').unlink()
            else:
                target=folder/('window0000.json'if fixture=='input_identity'else'results.json'if fixture=='summary_metric'else'parameters.json')
                data=json.loads(target.read_text());target.unlink()
                if fixture=='input_identity':data['input_sha256']='0'*64
                elif fixture=='summary_metric':data['NLL']+=1.
                elif fixture=='sample_budget':data['records'][0]['samples']=8192
                else:
                    row=next(r for r in data['records']if r['layout']=='transposed_matrix');row['layout']='storage_flat'
                target.write_text(json.dumps(data))
            try:inspect_case(fake,case,p,tokens,offsets,layout)
            except (AssertionError,FileNotFoundError)as error:results.append({'fixture':fixture,'rejected':True,'exception':type(error).__name__})
            else:raise AssertionError('Auditor accepted corrupted '+fixture)
    dest=OUT/'audit_rejection_fixtures_v1.json'
    dest.write_text(json.dumps({'status':'all_five_corrupted_copy_fixtures_rejected','source_sha256':sha(__file__),'auditor_source_sha256':sha(Path(__file__).with_name('audit_benchmark_v1.py')),
        'positive_case_PPL':baseline['PPL'],'fixtures':results,'scope':'Symlinks andcopiedJSON onlyintemporarydirectories; originalscientificartifactsuntouched.'},indent=2)+'\n')
    print('Five corrupted-copy fixtures rejected; original case passes.')

if __name__=='__main__':main()
