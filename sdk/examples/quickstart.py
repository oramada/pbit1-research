import numpy as np
from pbit1 import SinglePBit

weights = np.array([[-0.7, 0.2], [0.3, 0.8], [-0.1, 0.4]])
inputs = np.array([[1.0, -0.5, 0.25]])
codec = SinglePBit(eps=0)
encoded = codec.encode(weights)
output = codec.forward(inputs, encoded, 8192, np.random.default_rng(7))
print("Expected:", inputs @ weights)
print("Sampled: ", output)
