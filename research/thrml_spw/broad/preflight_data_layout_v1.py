"""No-text boundary/coverage fixtures for full-article evaluation layout."""
import hashlib,json
from pathlib import Path
from .data_layout_v1 import causal_windows,articles_from_rows

def main():
    sizes=[0,1,2,511,512,513,514,1023,1024,1025,1537,10000]
    for n in sizes:
        windows=causal_windows(n)
        targets=[i for w in windows for i in range(w['target_start'],w['target_stop'])]
        assert targets==list(range(1,n))
        assert all(w['context_start']<w['target_start'] and w['context_stop']-w['context_start']<=1024 for w in windows)
        assert all(w['target_stop']-w['target_start']<=512 for w in windows)
    rows=['','preamble',' = A = ','body',' == subsection == ','more',' = B = ','body2','']
    a=articles_from_rows(rows)
    assert len(a)==3 and [v['source_row_start'] for v in a]==[1,2,6]
    assert a[1]['text']=='\n'.join(rows[2:6])
    assert articles_from_rows(['',''])==[]
    out=Path('results/thrml_spw/broad_verification_20261105/additional_data_layout_v1_preflight')
    out.mkdir(exist_ok=False)
    files=[Path(__file__),Path(__file__).with_name('data_layout_v1.py')]
    report={'status':'all_target_coverage_context_bounds_and_article_boundary_fixtures_passed',
        'token_counts_tested':sizes,'target_policy':'Every token index1..N-1 within each article exactly once, context<=1024, targetstride<=512',
        'sources_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        'scope':'Synthetic layout only, no checkpoint, real corpus or quality scores.'}
    (out/'results.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))

if __name__=='__main__':main()
