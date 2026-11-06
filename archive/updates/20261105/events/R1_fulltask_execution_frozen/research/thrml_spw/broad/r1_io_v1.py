"""Stage-checkpoint layout for full-parameter reconstruction and evaluation."""
from pathlib import Path
import hashlib,json
from research.thrml_spw.lfm3.stage_checkpoint_v1 import StageStore

BASE=Path('results/thrml_spw/broad_verification_20261105')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(16*1024*1024),b''):h.update(block)
    return h.hexdigest()


def array_sha(values):
    h=hashlib.sha256();h.update(str(values.dtype).encode());h.update(json.dumps(list(values.shape)).encode())
    h.update(values.tobytes(order='C'));return h.hexdigest()


class CaseStore(StageStore):
    def _name(self,index,stage):
        if type(index)is not int or not 0<=index<100000 or stage not in {'parameter','evaluation'}:
            raise ValueError('Invalid fixed parameter/evaluation checkpoint identity')
        return self.root/f'{stage}_{index:05d}.json'


def program(path):
    import numpy as np
    from .parameter_program_v1 import validate
    with np.load(path,allow_pickle=False) as a:p={k:a[k].copy() for k in a.files}
    p['group_size']=int(p['group_size']);p['format_version']=int(p['format_version'])
    return validate(p)


def reconstructed(program,counts,samples):
    import numpy as np
    from .parameter_program_v1 import decode_probability
    c=np.asarray(counts)
    if c.ndim!=2 or c.shape[0] not in (1,4,8) or c.shape[1]!=len(program['codes']):
        raise ValueError('Complete declared bitplanes and coordinates required')
    if c.dtype.kind not in 'iu' or np.any(c>samples) or np.any(c<0):raise ValueError('Invalid counts')
    k=c.shape[0];numerator=np.sum(c.astype(np.int64)*(1<<np.arange(k,dtype=np.int64))[:,None],axis=0,dtype=np.int64)
    return decode_probability(program,numerator.astype(np.float64)/(samples*((1<<k)-1))).astype(np.float32)


def source_guard(protocol):
    for path,digest in protocol['sources_sha256'].items():
        if sha(path)!=digest:raise ValueError('Frozen source changed: '+path)
    for path,digest in protocol['inputs_sha256'].items():
        if sha(path)!=digest:raise ValueError('Frozen input changed: '+path)
    if protocol['status'] not in {'complete_R1_execution_and_statistics_frozen_before_scoring',
                                  'actual_model_synthetic_preflight_no_task_scoring'}:
        raise ValueError('Complete frozen execution contract required')
