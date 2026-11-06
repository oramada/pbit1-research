import unittest
from types import SimpleNamespace
import numpy as np
import torch
from .sequential_transformer import SequentialRunner,projection_targets

class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__();torch.manual_seed(19)
        self.embedding=torch.nn.Embedding(11,3)
        self.first=torch.nn.Linear(3,4)
        self.second=torch.nn.Linear(4,3)
        self.head=torch.nn.Linear(3,11)
    def forward(self,input_ids,use_cache=False):
        x=self.embedding(input_ids);x=torch.relu(self.first(x));x=self.second(x)
        return SimpleNamespace(logits=self.head(x))

class SequentialTransformerChecks(unittest.TestCase):
    def test_multi_projection_replay_and_regenerated_student_inputs(self):
        model=TinyModel().eval();runner=SequentialRunner(model,['first','second'])
        tokens=np.array([[1,2,3,4],[4,3,2,1]],dtype=np.int64)
        baseline=runner.evaluate(tokens)
        a=runner.evaluate(tokens,capture=0);b=runner.evaluate(tokens,capture=1)
        replay=runner.evaluate(tokens,{0:(a['inputs'],a['outputs']),1:(b['inputs'],b['outputs'])})
        np.testing.assert_array_equal(replay['nll'],baseline['nll'])
        self.assertEqual(replay['input_checks'],4)
        np.testing.assert_allclose(a['inputs']@runner.weights[0]+runner.biases[0],a['outputs'],atol=2e-7)
        modified=a['outputs']+.3
        student=runner.evaluate(tokens,{0:(a['inputs'],modified)},capture=1)
        np.testing.assert_array_equal(student['inputs'],np.maximum(modified,0))
        self.assertGreater(np.max(np.abs(student['inputs']-b['inputs'])),.1)
        final=runner.evaluate(tokens,{0:(a['inputs'],modified),1:(student['inputs'],student['outputs'])})
        np.testing.assert_array_equal(final['nll'],student['nll'])
        with torch.no_grad():model.first.weight.add_(1)
        runner.restore();np.testing.assert_array_equal(model.first.weight.detach().numpy().T,runner.weights[0])

    def test_explicit_targets_cover_three_blocks_in_forward_order(self):
        targets=projection_targets([0,1,2]);self.assertEqual(len(targets),12)
        self.assertEqual(targets[4],'transformer.h.1.attn.c_attn')
        self.assertEqual(targets[-1],'transformer.h.2.mlp.c_proj')
        with self.assertRaises(ValueError):SequentialRunner(TinyModel(),['first','first'])

if __name__=='__main__':unittest.main()
