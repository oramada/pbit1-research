"""Local-range native linear layer with finite, dimensionless bias controls."""
from .grouped_range_native import GroupedRangeNativeLinear
from .control_bank import ControlledProbabilityBank


class ControlledGroupedLinear(GroupedRangeNativeLinear):
    def __init__(self,weight,bias,bits,bound=4.,block_size=128):
        super().__init__(weight,bias,block_size)
        self.bank=ControlledProbabilityBank(self.bank.probabilities.reshape(self.bank.shape),bits,bound)
