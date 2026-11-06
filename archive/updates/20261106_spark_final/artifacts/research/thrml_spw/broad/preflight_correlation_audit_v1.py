"""Check independent audit rejects corrupted evidence, without new sampling."""
from pathlib import Path
import csv,json,shutil,tempfile
import numpy as np
from .verify_correlation_evidence_v1 import ROOT,audit,sha


def main():
    names=['manifest.json','dot_product_mse.csv','warmup_stationarity.csv','raw_native_trajectories_and_dot_means.npz']
    rejected=[]
    for corruption in ['reported_MSE','off_diagonal_covariance','missing_condition']:
        with tempfile.TemporaryDirectory(prefix='pbit1-correlated-audit-')as tmp:
            root=Path(tmp)
            for name in names:shutil.copyfile(ROOT/name,root/name)
            path=root/'dot_product_mse.csv'
            if corruption=='off_diagonal_covariance':
                path=root/names[-1]
                with np.load(path,allow_pickle=False)as raw:data={k:raw[k].copy()for k in raw.files}
                x=data['n4_J0.8_exact_cov_z_by_lag'];x[:,0,1]=0
                np.savez_compressed(path,**data)
            else:
                with path.open()as f:reader=csv.DictReader(f);columns=reader.fieldnames;rows=list(reader)
                if corruption=='reported_MSE':rows[0]['empirical_mse']='0.0'
                else:rows.pop()
                with path.open('w',newline='')as f:w=csv.DictWriter(f,fieldnames=columns);w.writeheader();w.writerows(rows)
            try:audit(root)
            except (AssertionError,ValueError):rejected.append(corruption)
            else:raise AssertionError('Corruption accepted: '+corruption)
    out=Path('results/thrml_spw/broad_verification_20261105/correlation_independent_audit_v1')
    p=out/'forgery_fixture.json';assert not p.exists()
    result={'status':'all_three_corrupted_evidence_fixtures_rejected','rejected':rejected,
        'source_sha256':sha(__file__),'auditor_sha256':sha(Path(__file__).with_name('verify_correlation_evidence_v1.py')),
        'scope':'Exercises evidence integrity assertions, not new Monte Carlo evidence.'}
    p.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':main()
