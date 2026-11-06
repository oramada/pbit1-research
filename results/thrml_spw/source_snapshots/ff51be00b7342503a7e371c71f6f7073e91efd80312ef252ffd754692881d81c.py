"""Closed-stage integrity checkpoints for one externally claimed experiment.

This is not a distributed lock. A Modal atomic job claim must ensure one live
writer. Completed stage metadata is atomically published only after its output
has closed and its hash has been checked. Orphan attempt files are retained.
"""
from pathlib import Path
import hashlib
import json
import os
import uuid

STAGES = {'BF16', 'same_fit_BF16', 'nominal_controls', 'native'}


class CheckpointIntegrityError(ValueError):
    pass


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(16*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


class StageStore:
    def __init__(self, root, contract):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.contract = contract
        self.contract_sha256 = hashlib.sha256(canonical(contract)).hexdigest()
        path = self.root/'contract.json'
        if path.exists():
            try: existing = json.loads(path.read_text())
            except (ValueError, OSError) as error:
                raise CheckpointIntegrityError('Unparseable existing contract') from error
            if canonical(existing) != canonical(contract):
                raise CheckpointIntegrityError('Checkpoint contract differs')
        else:
            self._atomic_json(path, contract)

    def _atomic_json(self, path, record):
        temp = path.with_name(path.name+'.partial-'+uuid.uuid4().hex)
        with temp.open('xb') as f:
            f.write(canonical(record))
            f.flush()
            os.fsync(f.fileno())
        if path.exists():
            raise FileExistsError(str(path))
        os.replace(temp, path)

    def _name(self, batch, stage):
        if type(batch) is not int or not 0 <= batch < 16 or stage not in STAGES:
            raise ValueError('Invalid frozen batch/stage')
        return self.root/f'stage_{batch}_{stage}.json'

    def new_attempt(self):
        path = self.root/'attempts'/uuid.uuid4().hex
        path.mkdir(parents=True, exist_ok=False)
        return path

    def _validate_record(self, record):
        if not {'output', 'metrics', 'module_calls', 'operator_calls'} <= set(record):
            raise CheckpointIntegrityError('Incomplete stage record')
        output = record['output']
        if set(output) != {'path', 'sha256'}:
            raise CheckpointIntegrityError('Output binding must contain path and SHA256')
        path = (self.root/output['path']).resolve()
        if not path.is_relative_to(self.root):
            raise CheckpointIntegrityError('Output escapes checkpoint namespace')
        if not path.is_file() or sha(path) != output['sha256']:
            raise CheckpointIntegrityError('Output is absent or has changed')
        canonical(record)  # Reject nonfinite/unserializable metadata.

    def commit(self, batch, stage, record):
        path = self._name(batch, stage)
        if path.exists():
            raise FileExistsError('A completed stage cannot be overwritten')
        self._validate_record(record)
        payload = {'version': 1, 'contract_sha256': self.contract_sha256,
            'batch': batch, 'stage': stage, 'record': record,
            'record_sha256': hashlib.sha256(canonical(record)).hexdigest()}
        self._atomic_json(path, payload)
        return self.load(batch, stage)

    def load(self, batch, stage):
        path = self._name(batch, stage)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
            if (data['version'] != 1 or data['batch'] != batch or data['stage'] != stage
                    or data['contract_sha256'] != self.contract_sha256
                    or data['record_sha256'] != hashlib.sha256(canonical(data['record'])).hexdigest()):
                raise CheckpointIntegrityError('Stage identity/metadata digest differs')
            self._validate_record(data['record'])
            return data['record']
        except (KeyError, TypeError, json.JSONDecodeError, OSError) as error:
            raise CheckpointIntegrityError('Invalid stage checkpoint') from error
