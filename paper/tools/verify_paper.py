"""Check provenance, key numbers, compilation diagnostics, and PDF text geometry."""
from pathlib import Path
import hashlib,json,re,subprocess,tempfile,xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads((ROOT/'evidence'/p).read_text())
def main():
 manifest=json.loads((ROOT/'EVIDENCE_MANIFEST.json').read_text())
 for entry in manifest:assert sha(ROOT/entry['paper_path'])==entry['sha256'],entry['paper_path']
 models=read('cross_model_summary_v2/results.json')['models']
 assert len(models)==3
 for m in models:
  assert m['seed_count']==5 and m['logical_pbits_per_learned_parameter']==1
  assert m['MC95_perplexity_ratio_interval'][1]<=1.02
 expected=[124439808,66955010,1728801536];assert [m['unique_parameters']for m in models]==expected
 budget=read('reduced_work_gpu_v1_audit.json')
 assert abs(budget['observation_fraction']-2378245865472/4077643628544)<1e-14
 assert budget['extra_range_metadata_bytes']==53088
 s=read('native_coarse_fractional_v1_audit.json')['summary']
 assert len(s)==6
 for layout in ['storage_flat','transposed_matrix']:
  group=[x for x in s if x['layout']==layout];assert len(group)==3
  assert group[1]['mean_MSE']<group[0]['mean_MSE']
  assert group[0]['node_observations']==group[1]['node_observations']
 ledger=read('resource_ledger_final_v1.json')
 for x in ledger['rows']:
  assert x['software_uint16_command_array_bytes']==2*x['unique_parameters']
  assert x['FP32_range_metadata_bytes']==8*x['range_groups']
  assert x['maximum_saved_GPU_allocation_bytes']<=2*1024**3
 report={'status':'passed','copied_input_hashes_checked':len(manifest),'numeric_checks':['three complete five-seed K1 cohorts','quality gates','lower-work budget','stronger native spatial baseline','four storage ledgers'],'pdfs':{}}
 for name in ['main','supplement']:
  pdf=ROOT/(name+'.pdf');log=(ROOT/(name+'.log')).read_text(errors='replace')
  bad=[line for line in log.splitlines()if 'Overfull' in line or ('Warning' in line and any(w in line.lower()for w in ['undefined','citation','reference']))]
  assert not bad,(name,bad)
  info=subprocess.check_output(['pdfinfo',str(pdf)],text=True);pages=int(re.search(r'Pages:\s+(\d+)',info).group(1))
  text=subprocess.check_output(['pdftotext',str(pdf),'-'],text=True)
  assert '??'not in text and 'Omar Ramadan'in text
  with tempfile.NamedTemporaryFile(suffix='.html')as f:
   subprocess.run(['pdftotext','-bbox',str(pdf),f.name],check=True)
   # Poppler emits control glyph codes for some TeX delimiters.
   xml=re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', Path(f.name).read_text())
   tree=ET.ElementTree(ET.fromstring(xml));ns={'x':'http://www.w3.org/1999/xhtml'}
   for page in tree.findall('.//x:page',ns):
    width=float(page.get('width'));height=float(page.get('height'))
    for word in page.findall('.//x:word',ns):
     assert 25<=float(word.get('xMin'))<float(word.get('xMax'))<=width-25,(name,'horizontal clipping',word.text)
     assert 15<=float(word.get('yMin'))<float(word.get('yMax'))<=height-15,(name,'vertical clipping',word.text)
  report['pdfs'][name]={'pages':pages,'sha256':sha(pdf),'bytes':pdf.stat().st_size,'unresolved_references':False,'overfull_boxes':False,'text_geometry_checked':True}
 out=ROOT/'verification';out.mkdir(exist_ok=True)
 (out/'automated_checks.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
