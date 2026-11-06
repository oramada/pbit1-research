"""Prepare a bounded, hash-verified local archive of accepted Spark evidence."""
from pathlib import Path
import datetime,gzip,hashlib,json,shutil,subprocess
from .prepare_v1 import OUT,sha


def main():
    repo=Path('publish/pbit1-experiments')
    root=repo/'updates/20261106_spark_final'
    root.mkdir(exist_ok=False)
    included=[]
    def copy(src):
        dest=root/'artifacts'/src
        if dest.exists():
            assert dest.read_bytes()==src.read_bytes()
            return
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(src,dest)
        included.append({'workspace_path':str(src),'archive_path':str(dest.relative_to(root)),'sha256':sha(src)})
    skip=set()
    names={'results.json','protocol.json','parameters.json','run.json','failure.json','replay_results.json'}
    omitted=[]
    for src in sorted(OUT.rglob('*')):
        if not src.is_file() or src.is_symlink():continue
        rel=src.relative_to(OUT)
        if rel.parts[0]in skip or src.name=='NEXT_ACTION.json':continue
        wanted=rel.as_posix()=='lfm3_error_covariance_v2/covariances.npz' or rel.parts[0]=='sdk_distribution_v2' or src.suffix in {'.md','.png','.svg','.xml'} or (src.suffix=='.json'and(len(rel.parts)==1 or src.name in names))
        if wanted:copy(src)
        elif src.suffix in {'.npz','.npy','.csv'}:
            omitted.append({'workspace_path':str(src),'sha256':sha(src),'bytes':src.stat().st_size})
    for src in sorted(Path('results/thrml_spw/source_snapshots').glob('*.py')):
        assert sha(src)==src.stem
        copy(src)
    for base in ['research','weight_encodings','models','operators','stochastic','analysis','configs']:
        source=Path(base)
        if not source.exists():continue
        for src in sorted(source.rglob('*')):
            if src.is_file()and not src.is_symlink()and src.suffix in {'.py','.yaml','.yml','.toml'}and'__pycache__'not in src.parts:copy(src)
    for base in ['publish/pbit1/src','publish/pbit1/tests']:
        for src in sorted(Path(base).rglob('*.py')):
            if'__pycache__'not in src.parts:copy(src)
    for src in sorted(Path('publish/pbit1').rglob('*')):
        if not src.is_file() or src.is_symlink():continue
        if any(x in {'.git','build','dist','__pycache__'}or x.endswith('.egg-info')for x in src.parts):continue
        if src.suffix in {'.py','.toml','.md','.svg','.png','.json','.txt'} or src.name in {'LICENSE','NOTICE','MANIFEST.in'}:copy(src)
    for name in ['SPARK_RESEARCH.md','master_design.tex']:
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
    index={'created_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'local_completed_research_snapshot_not_pushed','builder_sha256':sha(__file__),
      'accepted':['GPT2 full283225target5seedquality+allweight/36neuralreplays','DistilBERT full25000review5seedquality+allweight/36neuralreplays','LFM3 full294442target5seedquality+all7SDKweightpoints/21neuralreplays',
                  'LFM3 training7cases,55causalinterventions,baselineactivationstructure','NativeTHRMLhead,matched-observationandstrongcoarse/fractionalcontrols','Gaussianandillustrativedevice-calibrationcontrols','Lower-workGPT2fullcohortandall6point18windowreplays','LFM3conditionalcovarianceall48matricesaudited','GPT2cumulative41conditionsand6neuralreplays','SDKdevelopmentwheelinstalled23testsandnativeCLIpassed'],
      'excluded_running':[],
      'scope':'Sources and bounded summary/audit evidence. No checkpointweights, tokenarrays, fullneuraloutputs or rawsamplertrajectories. Fullraw-outputaudits require omittedassets or reproducible reruns. Preservedsourcesinclude planned/unexecuted scripts; acceptedclaims are limited to listed results.',
      'omitted_asset_count':len(omitted),'omitted_asset_bytes':sum(x['bytes']for x in omitted),'included':included}
    (root/'INDEX.json').write_text(json.dumps(index,indent=2)+'\n')
    (root/'README.md').write_text('# DGX Spark completed research snapshot · November 6\n\nThis update preserves the completed local investigation: three full model/task checks, causal and covariance diagnostics, matched-noise controls, stronger spatial controls, and the lower-sampling policy. It is prepared locally and has not been pushed.\n\n| Model/task | Accepted result |\n|---|---|\n| GPT-2 / 283225 WikiText2 targets | PPL ratio 1.00659; MC95 [0.99927, 1.01397] |\n| DistilBERT / 25000 IMDb reviews | Accuracy 87.4944% versus 87.512%; both gates pass |\n| LFM3 / 294442 WikiText2 targets | PPL ratio 1.00259; MC95 [0.98751, 1.01790] |\n| GPT-2 lower-sampling policy | 41.7% fewer nominal observations; PPL ratio 1.00780, upper MC95 1.01660 |\n\nAll tested neural representations retain one logical p-bit per learned scalar, with finite probability commands and shared range metadata. Whole-model tests reconstruct sampled weights and run conventional BF16 neural operators. Native THRML evidence is at operator scale. Universal numerical BF16 equality and physical energy savings have not been established.\n\n- [Consolidated research report](artifacts/SPARK_RESEARCH.md)\n- [Three-model summary](artifacts/results/thrml_spw/spark_depth_20261106/cross_model_summary_v2/results.json)\n- [Lower-work audit](artifacts/results/thrml_spw/spark_depth_20261106/reduced_work_gpu_v1_audit.json)\n- [Lower-work replay](artifacts/results/thrml_spw/spark_depth_20261106/reduced_work_replay_v1/results.json)\n- [Conditional covariance audit](artifacts/results/thrml_spw/spark_depth_20261106/lfm3_error_covariance_v2_audit.json)\n- [SDK build/install verification](artifacts/results/thrml_spw/spark_depth_20261106/sdk_build_verification_v2.json)\n\nRun `python verify_archive.py` here to check the snapshot. The source tree includes immutable content-addressed versions, and the SDK project and development distributions are included. Environment package versions are recorded under `environments/`.\n\nMost large raw arrays are omitted from Git and identified by hash in `OMITTED_ASSETS.json.gz`; checkpoint weights and dataset tokens must be restored under their own licenses. The compact covariance sufficient statistics are included. Raw-output audits require the omitted assets or complete reruns; summary JSON alone does not reproduce those audits. To rerun, copy the relevant source trees from `artifacts/` into a fresh project workspace and restore assets using the pinned preparation protocols. Existing output directories use exclusive creation or frozen contracts and should not be overwritten.\n\nThe earlier core snapshot and original root history remain intact. Planned and failed sources are retained for provenance; their existence does not establish completed experiments. No cloud execution or physical hardware access is part of this snapshot.\n')
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
    if '## Latest accepted Spark evidence' in text:
        a=text.index('## Latest accepted Spark evidence');b=text.index('## Earlier adapted LFM3 result');text=text[:a]+text[b:]
    section='## Latest completed Spark investigation\n\nThe [November 6 final research update](updates/20261106_spark_final/README.md) contains three complete model/task quality checks, independent replays, layer/covariance studies, and a verified 41.7% reduction in nominal sampling work. The [earlier core snapshot](updates/20261106_spark_core/README.md) is preserved. These are scoped software results; universal BF16 numerical equality and physical energy advantages are not established.\n\n'
    pos=text.index('## Earlier adapted LFM3 result');readme.write_text(text[:pos]+section+text[pos:])
    root_manifest=json.loads((repo/'FILES.sha256.json').read_text());root_manifest['README.md']=sha(readme)
    for f in root.rglob('*'):
        if f.is_file():root_manifest[str(f.relative_to(repo))]=sha(f)
    (repo/'FILES.sha256.json').write_text(json.dumps(root_manifest,indent=2)+'\n')
    subprocess.run(['python3',str(repo/'tools/verify_snapshot.py')],check=True)
    print(json.dumps({'archive':str(root),'included_sources_and_reports':len(included),'omitted_assets':len(omitted),'omitted_bytes':sum(x['bytes']for x in omitted)}))

if __name__=='__main__':main()
