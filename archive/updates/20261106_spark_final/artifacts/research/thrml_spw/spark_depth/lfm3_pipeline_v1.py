"""Local-only serial continuation: finish training, audit, intervene, then qualify test."""
from pathlib import Path
import json,os,subprocess,sys,time
from .prepare_v1 import OUT,sha
from .benchmark_v1 import atomic_json

WAIT_PID=2940519

def main():
 root=OUT/'lfm3_pipeline_v1';root.mkdir(exist_ok=False);begin=time.monotonic()
 scripts=['analyze_lfm3_training_depth_v1','lfm3_causal_depth_v2','analyze_lfm3_causal_v1','lfm3_benchmark_v2']
 frozen={n:sha(Path(__file__).with_name(n+'.py'))for n in scripts}
 initial=Path(f'/proc/{WAIT_PID}/cmdline')
 if initial.exists():assert b'research.thrml_spw.spark_depth.lfm3_training_depth_v1'in initial.read_bytes().split(b'\0')
 atomic_json(root/'protocol.json',{'source_sha256':sha(__file__),'wait_pid':WAIT_PID,'scripts':frozen,'cloud':'disabled','serial_GPU':True,'scope':'Local continuation only. No goal completion or publication.'})
 env=os.environ.copy();env.update({'CUBLAS_WORKSPACE_CONFIG':':4096:8','OPENBLAS_NUM_THREADS':'2','OMP_NUM_THREADS':'2','HF_HUB_DISABLE_PROGRESS_BARS':'1'})
 def status(stage,**kw):atomic_json(root/'progress.json',{'stage':stage,'elapsed_seconds':time.monotonic()-begin,**kw})
 try:
  while initial.exists():
   args=initial.read_bytes().split(b'\0')
   if b'research.thrml_spw.spark_depth.lfm3_training_depth_v1'not in args:break
   status('waiting_for_live_training',pid=WAIT_PID);time.sleep(10)
  assert(OUT/'lfm3_training_depth_v1/results.json').exists(),'Training exited without a complete cohort'
  for script in scripts:
   assert sha(Path(__file__).with_name(script+'.py'))==frozen[script],'Queued source changed'
   if script=='lfm3_benchmark_v2':
    training=json.loads((OUT/'lfm3_training_depth_v1/results.json').read_text())
    if training['MC95_t4_PPL_ratio_interval'][1]>1.02:
     status('training_gate_failed_no_heldout_run');return
   python='.venv-thrml/bin/python'if script.startswith('analyze_')else'.venv/bin/python'
   status('running',script=script);print('START',script,flush=True)
   proc=subprocess.Popen([python,'-u','-m','research.thrml_spw.spark_depth.'+script],env=env)
   while proc.poll()is None:status('running',script=script,pid=proc.pid);time.sleep(10)
   assert proc.returncode==0,(script,proc.returncode)
   print('DONE',script,flush=True)
  status('pipeline_finished_benchmarks_still_require_independent_audit')
 except Exception as error:
  atomic_json(root/'failure.json',{'type':type(error).__name__,'message':str(error)});raise
if __name__=='__main__':main()
