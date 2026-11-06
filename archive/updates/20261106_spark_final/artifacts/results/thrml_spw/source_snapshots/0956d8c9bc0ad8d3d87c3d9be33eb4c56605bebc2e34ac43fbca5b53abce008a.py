"""Metadata-only local gate watcher; never retries a sampled model case."""
from pathlib import Path
import json
import subprocess
import time
import modal


def main():
    root = Path('results/thrml_spw/modal_h100_v1')
    destination = root / 'holdout_assets_v1.json'
    deadline = time.time() + 24 * 3600
    while not destination.exists():
        result = subprocess.run(['.venv/bin/python', '-m', 'cloud.modal_h100_v1.freeze_holdout_v2'])
        if result.returncode == 0:
            break
        if result.returncode != 2:
            raise RuntimeError('Eligibility did not authorize holdout: ' + str(result.returncode))
        assert time.time() < deadline, 'Validation gate watcher deadline'
        time.sleep(30)
    manifest = json.loads(destination.read_text())
    volume = modal.Volume.from_name('spw-h100-validation-v1')
    names = list(manifest['files'])
    for start in range(0, len(names), 24):
        with volume.batch_upload(force=True) as batch:
            for name in names[start:start+24]:
                batch.put_file(name, '/' + name)
        print('heldout_input_metadata_uploaded', min(start+24, len(names)), flush=True)
    with volume.batch_upload(force=True) as batch:
        batch.put_file(destination, '/holdout_assets_v1.json')
    print('complete_frozen_heldout_inputs_released', flush=True)


if __name__ == '__main__':
    main()
