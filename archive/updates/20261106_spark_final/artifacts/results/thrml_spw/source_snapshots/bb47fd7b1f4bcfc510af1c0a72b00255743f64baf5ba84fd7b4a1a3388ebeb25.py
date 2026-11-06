"""Prepare a bounded, hash-verified local archive of accepted Spark evidence."""
from pathlib import Path
import datetime,gzip,hashlib,json,shutil,subprocess
from .prepare_v1 import OUT,sha


def main():
    repo=Path('publish/pbit1-experiments')
    root=repo/'updates/20261106_spark_core'
    root.mkdir(exist_ok=False)
    included=[]
    def copy(src):
        dest=root/'artifacts'/src
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(src,dest)
        included.append({'workspace_path':str(src),'archive_path':str(dest.relative_to(root)),'sha256':sha(src)})
    skip={'reduced_work_gpu_v1','reduced_work_replay_v1','lfm3_error_covariance_v1','lfm3_error_covariance_v2','post_reduced_work_v1'}
    names={'results.json','protocol.json','parameters.json','run.json','failure.json','replay_results.json'}
    omitted=[]
    for src in sorted(OUT.rglob('*')):
        if not src.is_file() or src.is_symlink():continue
        rel=src.relative_to(OUT)
        if rel.parts[0]in skip or src.name=='NEXT_ACTION.json':continue
        wanted=src.suffix in {'.md','.png','.svg'} or (src.suffix=='.json'and(len(rel.parts)==1 or src.name in names))
        if wanted:copy(src)
        elif src.suffix in {'.npz','.npy','.csv'}:
            omitted.append({'workspace_path':str(src),'sha256':sha(src),'bytes':src.stat().st_size})
    for base in ['research','weight_encodings','models','operators','stochastic','analysis','configs']:
        source=Path(base)
        if not source.exists():continue
        for src in sorted(source.rglob('*')):
            if src.is_file()and not src.is_symlink()and src.suffix in {'.py','.yaml','.yml','.toml'}and'__pycache__'not in src.parts:copy(src)
    for base in ['publish/pbit1/src','publish/pbit1/tests']:
        for src in sorted(Path(base).rglob('*.py')):
            if'__pycache__'not in src.parts:copy(src)
    for name in ['publish/pbit1/pyproject.toml','publish/pbit1/README.md','SPARK_RESEARCH.md','master_design.tex']:
        src=Path(name)
        if src.exists():copy(src)
    with gzip.open(root/'OMITTED_ASSETS.json.gz','wt')as f:json.dump(omitted,f)
    history=root/'history';history.mkdir()
    for name in ['README.md','FILES.sha256.json','SNAPSHOT.json']:shutil.copyfile(repo/name,history/('ROOT_'+name))
    env=root/'environments';env.mkdir()
    script="import importlib.metadata as m,json,platform,sys; print(json.dumps({'python':sys.version,'platform':platform.platform(),'distributions':dict(sorted((d.metadata['Name'],d.version)for d in m.distributions() if d.metadata['Name']))},indent=2))"
    for name,python in [('torch_runtime','.venv/bin/python'),('thrml_runtime','.venv-thrml/bin/python')]:
        data=subprocess.check_output([python,'-c',script],text=True)
        (env/(name+'.json')).write_text(data)
    index={'created_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'local_accepted_core_snapshot_not_pushed','builder_sha256':sha(__file__),
      'accepted':['GPT2 full283225target5seedquality+allweight/36neuralreplays','DistilBERT full25000review5seedquality+allweight/36neuralreplays','LFM3 full294442target5seedquality+all7SDKweightpoints/21neuralreplays',
                  'LFM3 training7cases,55causalinterventions,baselineactivationstructure','NativeTHRMLhead,matched-observationandstrongcoarse/fractionalcontrols','Gaussianandillustrativedevice-calibrationcontrols'],
      'excluded_running':['reduced_work_gpu_v1','lfm3_error_covariance_v2'],
      'scope':'Sources and bounded summary/audit evidence. No checkpointweights, tokenarrays, fullneuraloutputs or rawsamplertrajectories. Fullraw-outputaudits require omittedassets or reproducible reruns. Preservedsourcesinclude planned/unexecuted scripts; acceptedclaims are limited to listed results.',
      'omitted_asset_count':len(omitted),'omitted_asset_bytes':sum(x['bytes']for x in omitted),'included':included}
    (root/'INDEX.json').write_text(json.dumps(index,indent=2)+'\n')
    (root/'README.md').write_text('''# DGX Spark accepted core evidence · November 6

This bounded snapshot preserves three complete model/task quality checks, their audits and replays, and the layer/control studies completed with them. It is a local working-tree archive update; it has not been pushed.

| Model/task | Accepted result |
|---|---|
| GPT-2 / 283225 WikiText2 targets | Perplexity ratio1.00659; MC95[0.99927,1.01397] |
| DistilBERT / 25000 IMDb reviews | Accuracy87.4944% vs87.512%; both frozen gates pass |
| LFM3 / 294442 WikiText2 targets | Perplexity ratio1.00259; MC95[0.98751,1.01790] |

All use one logical p-bit per learned scalar and finite12bit probability commands. These are independent-binomial R1 reconstructions followed by conventional BF16 GPU execution. Native THRML evidence is at operator scale. The results establish neither universal numerical BF16 equality nor measured physical energy savings. The stronger four-element coarse/fractional control achieves lower operator error with more elements and stored information, so the density/time tradeoff remains explicit.

- [Three-model summary](artifacts/results/thrml_spw/spark_depth_20261106/cross_model_summary_v2/results.json)
- [LFM3 complete score/coverage audit](artifacts/results/thrml_spw/spark_depth_20261106/lfm3_benchmark_audit_v1/results.json)
- [LFM3 independent SDK replay](artifacts/results/thrml_spw/spark_depth_20261106/lfm3_benchmark_replay_v1/results.json)
- [Strong spatial control audit](artifacts/results/thrml_spw/spark_depth_20261106/native_coarse_fractional_v1_audit.json)
- [Workspace research report](artifacts/SPARK_RESEARCH.md) (its original links are workspace-relative)

The lower-work GPU test and conditional error-covariance study were still running/queued at this snapshot cutoff. Their result namespaces are excluded. Sources for planned studies are retained but are not evidence of execution.

Run `python verify_archive.py` here to check every included byte. `INDEX.json` records source paths and hashes; `OMITTED_ASSETS.json.gz` records hashes of raw arrays omitted from Git. Checkpoints and dataset tokens must be obtained under their own licenses. Copy the archived source trees from `artifacts/` into a fresh project workspace, restore the pinned assets named by each preparation protocol, and use the recorded environment versions and command modules. The raw-output audits require the omitted data or a complete rerun; summary JSON alone cannot reproduce them.

Historical root README, snapshot descriptor and hash list are retained under `history/`. Cloud scripts are historical sources; this update authorizes no cloud execution. All new work here used the local GB10 with a2GiB allocator cap.
''')
    (root/'verify_archive.py').write_text('''from pathlib import Path
import hashlib,json
root=Path(__file__).resolve().parent
manifest=json.loads((root/'FILES.sha256.json').read_text())
actual={str(p.relative_to(root))for p in root.rglob('*')if p.is_file()and p.name!='FILES.sha256.json'}
assert actual==set(manifest), 'Missing or unexpected archive files'
for name,expected in manifest.items():
 h=hashlib.sha256()
 with(root/name).open('rb')as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 assert h.hexdigest()==expected,name
print('Verified',len(manifest),'archive file hashes')
''')
    manifest={str(p.relative_to(root)):sha(p)for p in sorted(root.rglob('*'))if p.is_file()}
    (root/'FILES.sha256.json').write_text(json.dumps(manifest,indent=2)+'\n')
    subprocess.run(['python3',str(root/'verify_archive.py')],check=True)
    # Preserve the older root documents in the dated update before adding its link.
    readme=repo/'README.md';text=readme.read_text().replace('## Main result','## Earlier adapted LFM3 result',1).replace('This public snapshot','This bounded snapshot')
    section='''## Latest accepted Spark evidence

The [November6 core update](updates/20261106_spark_core/README.md) adds complete, audited GPT-2, DistilBERT and LFM3 model/task checks, all-parameter encodings, independent neural replays, layer diagnostics, nonideal controls and a stronger native THRML spatial baseline. All three tested quality criteria pass. These remain scoped software results; universal BF16 numerical equality and physical energy advantages are not established. Later sampling-cost and error-covariance studies were pending at this snapshot cutoff.

'''
    pos=text.index('## Earlier adapted LFM3 result');readme.write_text(text[:pos]+section+text[pos:])
    root_manifest=json.loads((repo/'FILES.sha256.json').read_text());root_manifest['README.md']=sha(readme)
    for f in root.rglob('*'):
        if f.is_file():root_manifest[str(f.relative_to(repo))]=sha(f)
    (repo/'FILES.sha256.json').write_text(json.dumps(root_manifest,indent=2)+'\n')
    subprocess.run(['python3',str(repo/'tools/verify_snapshot.py')],check=True)
    print(json.dumps({'archive':str(root),'included_sources_and_reports':len(included),'omitted_assets':len(omitted),'omitted_bytes':sum(x['bytes']for x in omitted)}))

if __name__=='__main__':main()
