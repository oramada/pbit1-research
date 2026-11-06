"""Read-only point-in-time GPU/process memory snapshot on an existing job.

No extra GPU allocation, sampler calls, process changes or inference. Process
RSS is not unique total resident memory; NVML is a point sample, not a peak.
"""
from pathlib import Path
import datetime,hashlib,json,subprocess

SCRIPT=r'''
import datetime,json,os,subprocess
from pathlib import Path
allowed={'research.thrml_spw.broad.highT_case_v1','research.thrml_spw.broad.highT_worker_v1','research.thrml_spw.gpu_environment'}
processes=[]
for p in Path('/proc').glob('[0-9]*'):
 try:
  args=[x.decode()for x in (p/'cmdline').read_bytes().split(b'\0')if x]
  if '-m'not in args:continue
  module=args[args.index('-m')+1]
  if module not in allowed:continue
  fields=dict(line.split(':',1)for line in (p/'status').read_text().splitlines()if ':'in line)
  record={'pid':int(p.name),'module':module,'state':fields.get('State','').strip(),'VmRSS':fields.get('VmRSS','').strip(),'VmHWM':fields.get('VmHWM','').strip()}
  if '--case'in args:record['case']=args[args.index('--case')+1]
  processes.append(record)
 except (OSError,ValueError,IndexError):pass
raw=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,utilization.gpu,memory.used,memory.total,power.draw','--format=csv,noheader,nounits'],text=True)
print(json.dumps({'observed_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'GPU_query_fields':['name','uuid','utilization_pct','memory_used_MiB','memory_total_MiB','power_draw_W'],'GPU_query_raw':raw.strip(),'processes':processes}))
'''


def main():
    container='ta-01M6Q1V04WFTZ2V8E80FCW1QXR'
    command=['uvx','--python','.venv/bin/python','--from','modal','modal','container','exec','--no-pty','--',container,'python','-c',SCRIPT]
    result=subprocess.run(command,text=True,capture_output=True,check=True)
    observation=json.loads(result.stdout)
    assert len(observation['processes'])>=2
    fields=[x.strip()for x in observation['GPU_query_raw'].split(',')]
    assert fields[0].startswith('NVIDIA H100') and float(fields[2])>0
    record={'status':'existing_highT_H100_readonly_live_resource_snapshot','source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'container_id':container,'observed':observation,'instrumentation_stderr':result.stderr,
        'scope':'Onepoint during ongoing nativeweight reconstruction. GPUmemory is allprocesses onthatGPU; processRSS/HWMareindividual andmaydoublecountsharedpages. This is not fullpipelinepeakRAM, one-bit GPUcompression, a timingcomparison, inferenceenergy or physicalpbitpower. Sidecar only, no frozenworker/program/math change or new sampling.'}
    out=Path('results/thrml_spw/broad_verification_20261105/live_resource_snapshot_v1');out.mkdir(exist_ok=False)
    (out/'results.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))

if __name__=='__main__':main()
