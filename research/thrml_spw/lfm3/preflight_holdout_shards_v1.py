"""Coverage/key/contract fixtures before any sharded native work."""
from pathlib import Path
import copy,json,hashlib
import numpy as np
from .holdout_shard_contract_v1 import validate_subset,sha
from .holdout_batch_layout import assemble_losses


def main():
    root=Path('results/thrml_spw/broad_verification_20261105/lfm3_sharding_preparation_v1')
    p=json.loads(Path('results/thrml_spw/lfm3_1p7b/holdout_native_v4_protocol.json').read_text());plan=p['batch_plan']
    assignments={str(i):list(range(i,16,4))for i in range(4)}
    assert sorted(i for ids in assignments.values()for i in ids)==list(range(16))
    records=[{'batch_index':i,'token_NLL':np.full(plan[i]['loss_shape'],i+.125).tolist()}for i in range(16)]
    assembled=[];checks=[]
    for name,indices in assignments.items():
        rr=[records[i]for i in reversed(indices)];validate_subset(plan,indices,rr);assembled.extend(rr)
        assert sum(plan[i]['global_target_stop']-plan[i]['global_target_start']for i in indices)==2560
        for i in indices:
            assert [287+i*287+j for j in range(287)]==list(range(287+i*287,287+(i+1)*287))
    x=assemble_losses(plan,records);y=assemble_losses(plan,assembled)
    for a,b in zip(x,y):np.testing.assert_array_equal(a,b)
    checks+=['all16globalbatches_exactlyonce','all10240targets_identical_after_permuted4shard_merge','global_callkeys_unchanged','each_shard2560targets']
    indices=assignments['0'];valid=[records[i]for i in indices]
    for name,bad in [('missing',valid[:-1]),('duplicate',valid[:-1]+[valid[0]]),('wrong_batch',[records[1],*valid[1:]]),('wrong_shape',[{'batch_index':0,'token_NLL':[[0.]]},*valid[1:]])]:
        try:validate_subset(plan,indices,bad)
        except ValueError:checks.append('reject_'+name)
        else:raise AssertionError('Corrupted shard accepted')
    r={'status':'batch_shard_globalcoverage_callkey_and_corruption_fixtures_passed','source_sha256':sha(__file__),
        'contract_source_sha256':sha(Path(__file__).with_name('holdout_shard_contract_v1.py')),'checks':checks,'assignments':assignments,
        'scope':'Pure synthetic loss/coverage/key fixtures, no neuralforward/native sampling or authorization to replace the active transport. Actual closedbatchreplay and full merge/audit gates remain.'}
    with (root/'coverage_preflight.json').open('x')as f:json.dump(r,f,indent=2);f.write('\n')
    print(json.dumps(r))

if __name__=='__main__':main()
