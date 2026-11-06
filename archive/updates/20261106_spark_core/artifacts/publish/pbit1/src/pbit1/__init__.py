"""One programmable p-bit per parameter, with temporal precision."""
from .encodings.single_pbit import SinglePBit, SinglePBitWeights

__version__ = "0.1.0a2.dev0"
__all__ = ["SinglePBit", "SinglePBitWeights", "NativeProbabilityBank"]

def __getattr__(name):
    if name == "NativeProbabilityBank":
        from .native import NativeProbabilityBank
        return NativeProbabilityBank
    raise AttributeError(name)

from .program import FinitePBitProgram, Reconstruction
__all__ += ["FinitePBitProgram", "Reconstruction"]
