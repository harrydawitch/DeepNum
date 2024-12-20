import numpy as np
from Components.Layers.Base import Layer
 

class Flatten(Layer):
    def __init__(self):
        super().__init__()
        self.name = 'Flatten'

    def forward(self, inputs):
        self.data_in = inputs.shape
        self.data_out = (self.data_in[0], -1)
        
        out = inputs.ravel().reshape(self.data_out)
        self.data_out = self.data_out[1]
        
        return out

    def backward(self, dout):

        dout = dout.reshape(self.data_in) 
        return dout  
