import numpy as np
from Components.Layers.Base import Layer
 

class Flatten(Layer):
    def __init__(self):
        super().__init__()
        self.name = 'Flatten'

    def forward(self, inputs):
        N, C, H, W = inputs.shape
        self.data_in = inputs.shape

        # The input shape from previous layer follow the channel first convention (NCHW)
        # Because my next layer which is Dense need input to be [features, batches]
        # So i reshape to be (C * H * W, N)
        flattened_output = inputs.reshape(C * H * W , N)  
        return flattened_output

    def backward(self, dout):

        dout_reshaped = dout.reshape(self.data_in) 
        return dout_reshaped  
