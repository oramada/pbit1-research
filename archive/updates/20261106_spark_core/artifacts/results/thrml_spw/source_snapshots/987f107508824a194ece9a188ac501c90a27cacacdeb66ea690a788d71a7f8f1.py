import json,tempfile,unittest
from pathlib import Path
import numpy as np
from .block_control_worker import Worker as OriginalWorker
from .block_cached_control_worker import Worker as CachedWorker

class WorkerCacheChecks(unittest.TestCase):
    def test_twelve_operators_keep_their_compiled_functions_and_exact_outputs(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);master=root/'weights.npz';x=root/'x.npy';ref=root/'ref.npy'
            weight=np.array([[-.2,.1],[.1,.3],[.5,.8]],dtype=np.float32);bias=np.zeros(2,dtype=np.float32)
            np.savez(master,**{f'weight_{i}':weight for i in range(12)},**{f'bias_{i}':bias for i in range(12)})
            inputs=np.array([[[.2,-.5,.7]]],dtype=np.float32);np.save(x,inputs);np.save(ref,inputs@weight)
            old,new=OriginalWorker(),CachedWorker();old_first=None;new_first=None
            for stage in range(12):
                request={'weights':str(master),'inputs':str(x),'reference':str(ref),'stage':stage,'samples':2,'seed':0,'domain':17,'contexts_per_batch':8,'field_bits':8,'field_bound':4.}
                for name,worker in [('old',old),('new',new)]:
                    response=worker.execute({**request,'output':str(root/f'{name}{stage}.npy')});self.assertEqual(response['status'],'ok')
                np.testing.assert_array_equal(np.load(root/f'old{stage}.npy'),np.load(root/f'new{stage}.npy'))
                if stage==0:
                    old_first=list(old.banks.values())[0].compiled(2)
                    new_first=list(new.compiled_functions.values())[0]
            self.assertIsNot(old_first,list(old.banks.values())[0].compiled(2))
            self.assertEqual(len(new.compiled_functions),12)
            self.assertIs(new_first,list(new.compiled_functions.values())[0])
            request.update(stage=0,output=str(root/'new0_repeat.npy'))
            new.execute(request)
            self.assertIs(new_first,list(new.compiled_functions.values())[0])
            np.testing.assert_array_equal(np.load(root/'new0.npy'),np.load(root/'new0_repeat.npy'))
            old_meta=json.loads((root/'old0.json').read_text());new_meta=json.loads((root/'new0.json').read_text())
            self.assertEqual(old_meta['free_node_updates'],new_meta['free_node_updates'])

if __name__=='__main__':unittest.main()
