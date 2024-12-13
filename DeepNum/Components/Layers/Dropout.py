import numpy as np
from Components.Layers.Base import Layer


class Dropout(Layer):
    def __init__(self, keep_rate= 0):

        if not 0 < keep_rate < 1:
            raise ValueError('Rate must be above 0 and below 1')
        super().__init__()
        self.rate = keep_rate
        self.name = 'Dropout'




    def _init_mask(self, inputs):
        n, m  = inputs.shape
        self.mask = (np.random.rand(n, m) < self.rate).astype(int)



    def forward(self, inputs):

        self._init_mask(inputs)

        data_out = inputs * self.mask / self.rate
        
        return data_out
    


    def backward(self, dout):
        dout = dout * self.mask / self.rate
        return dout