import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core.NeuralNetwork import Model
from Core.Layers import Dense, BatchNormalization, Dropout, Conv2D
from Core.Optimizers import Adam, RMSprop, Momentum,Gradient_Descent
from Core.Activations import ReLU, Softmax
from Core.Regularizers import L1, L2
from Core.Losses import categorical_crossentropy



import numpy as np

# Define a test case with simple data and a backward pass
input_data = np.random.randn(1, 5, 5, 3)  # (N=1, H=5, W=5, C=3)
filters = np.random.randn(3, 3, 3, 2)  # (filter_h=3, filter_w=3, in_channels=3, out_channels=2)
bias = np.zeros(2)  # Bias for 2 output channels
stride = 1
padding = 0

conv_layer = Conv2D(n_filters=2, filter_size=(3, 3), stride=stride, padding=padding, activation= 'relu')
conv_layer.input = 3  # Number of input channels
conv_layer.init_params()

# Forward pass
output = conv_layer.forward(input_data)

# Define a simple gradient from the next layer (e.g., just a ones matrix)
dout = np.ones_like(output)  # Same shape as the output

# Backward pass
dX = conv_layer.backward(dout)
print("dX shape:", dX.shape)  # Expected: (1, 5, 5, 3) (same as input)
print("dw shape:", conv_layer.dw.shape)  # Expected: (3, 3, 3, 2) (same as filter weights)
print("db shape:", conv_layer.db.shape)  # Expected: (2,) (same as bias size)

# Check gradients for correctness (you can perform numerical checks or assert expected shapes)
assert dX.shape == (1, 5, 5, 3), f"dX shape mismatch: {dX.shape}"
assert conv_layer.dw.shape == (3, 3, 3, 2), f"dw shape mismatch: {conv_layer.dw.shape}"
assert conv_layer.db.shape == (2,), f"db shape mismatch: {conv_layer.db.shape}"

