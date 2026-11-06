"""Continue serially to the prespecified LFM3 error-covariance measurement."""
from pathlib import Path
import json, os, subprocess, time
from .prepare_v1 import OUT, sha
from .benchmark_v1 import atomic_json


def main():
    root = OUT / 'post_reduced_work_v1'
    root.mkdir(exist_ok=False)
    scripts = ['lfm3_error_covariance_v2', 'audit_lfm3_error_covariance_v1']
    frozen = {name: sha(Path(__file__).with_name(name+'.py')) for name in scripts}
    atomic_json(root/'protocol.json', {'source_sha256': sha(__file__), 'wait_pid': 2973810, 'scripts': frozen, 'cloud': 'disabled'})
    observed = Path('/proc/2973810/cmdline')
    begin = time.monotonic()
    try:
        while True:
            try:
                active = b'research.thrml_spw.spark_depth.reduced_work_gpu_v1' in observed.read_bytes().split(b'\0')
            except FileNotFoundError:
                active = False
            if not active:
                break
            atomic_json(root/'progress.json', {'stage': 'waiting_for_reduced_work_GPU', 'elapsed_seconds': time.monotonic()-begin})
            time.sleep(10)
        assert (OUT/'reduced_work_gpu_v1/results.json').exists(), 'Reduced-work job exited without a complete cohort'
        for name in scripts:
            assert sha(Path(__file__).with_name(name+'.py')) == frozen[name]
            python = '.venv-thrml/bin/python' if name.startswith('audit_') else '.venv/bin/python'
            atomic_json(root/'progress.json', {'stage': 'running', 'script': name, 'elapsed_seconds': time.monotonic()-begin})
            print('START', name, flush=True)
            subprocess.run([python, '-u', '-m', 'research.thrml_spw.spark_depth.'+name], check=True, env=os.environ.copy())
            print('DONE', name, flush=True)
        atomic_json(root/'progress.json', {'stage': 'complete', 'elapsed_seconds': time.monotonic()-begin})
    except Exception as error:
        atomic_json(root/'failure.json', {'type': type(error).__name__, 'message': str(error)})
        raise

if __name__ == '__main__':
    main()
