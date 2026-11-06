"""Deterministic archive restoration, no stochastic experiments."""
from pathlib import Path
import modal
app=modal.App('spw-h100-input-restoration-v1')
volume=modal.Volume.from_name('spw-h100-validation-v1')
image=modal.Image.from_id('im-JBjlQBiXCDO9RMFnms6ffX')
@app.function(image=image,cpu=4,memory=16384,volumes={'/workspace':volume},timeout=1800,retries=0)
def restore(script: str):
    import subprocess
    result=subprocess.run(['/opt/torch/bin/python','-c',script],capture_output=True,text=True)
    print(result.stdout,flush=True)
    print(result.stderr[-6000:],flush=True)
    volume.commit()
    assert result.returncode==0,result.returncode
    return result.stdout
@app.local_entrypoint()
def main():
    print(restore.remote(Path('cloud/modal_h100_v1/rebuild_words.py').read_text()))
