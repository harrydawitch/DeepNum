import numpy as np

class Dropout:
    def __init__(self, rate= 0):
        if not 0 < rate < 1:
            raise ValueError('Rate must be above 0 and below 1')
        
        self.rate = rate
        self.name = 'Dropout'
        self.units = None
        self.learnable = False
        
    
    def _init_mask(self, inputs):
        n, m = inputs.shape
        self.mask = (np.random.rand(n, m) < self.rate).astype(int)

    def forward(self, inputs):

        self._init_mask(inputs)

        outputs = inputs * self.mask / self.rate
        return outputs
    
    def backward(self, da, y):
        da = da * self.mask / self.rate
        return da