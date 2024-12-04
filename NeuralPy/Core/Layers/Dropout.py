import numpy as np

class Dropout:
    def __init__(self, keep_rate= 0):
        if not 0 < keep_rate < 1:
            raise ValueError('Rate must be above 0 and below 1')
        
        self.rate = keep_rate
        self.name = 'Dropout'
        self._training_= True
        
    
    def _init_mask(self, inputs):
        n, m = inputs.shape
        self.mask = (np.random.rand(n, m) < self.rate).astype(int)

    def forward(self, inputs):

        self._init_mask(inputs)

        outputs = inputs * self.mask / self.rate
        return outputs
    
    def backward(self, dout):
        dout = dout * self.mask / self.rate
        return dout