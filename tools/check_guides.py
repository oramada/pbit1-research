"""Check current guide links and overview data, leaving frozen archives intact."""
from pathlib import Path
import hashlib,json,re
root=Path(__file__).resolve().parents[1]
guides=[root/'README.md',root/'REPRODUCING.md',root/'docs/OVERVIEW.md',
        root/'docs/EVIDENCE.md',root/'paper/README.md',root/'provenance/README.md']
count=0
for p in guides:
    text=re.sub(r'```.*?```','',p.read_text(),flags=re.S)
    links=re.findall(r'\]\(([^)]+)\)',text)+re.findall(r'(?:src|srcset)="([^"]+)"',text)
    for link in links:
        if link.startswith(('http:','https:','mailto:','#')):continue
        target=link.split('#')[0]
        if target:
            assert (p.parent/target).exists(),f'{p.relative_to(root)}: {link}'
            count+=1
chart=json.loads((root/'docs/assets/overview_data.json').read_text())
source=root/chart['source']
assert hashlib.sha256(source.read_bytes()).hexdigest()==chart['source_sha256']
models=json.loads(source.read_text())['models']
assert len(chart['rows'])==len(models)==3
for row,model in zip(chart['rows'],models):
    if model['model']=='DistilBERT':
        assert row['bf16']==100*model['BF16_accuracy']
        assert row['pbit1']==100*model['mean_selected_accuracy']
    else:
        assert row['bf16']==model['BF16_PPL']
        assert row['pbit1']==model['selected_PPL_at_mean_NLL']
print(f'Checked {count} local guide targets and all three overview score rows.')
