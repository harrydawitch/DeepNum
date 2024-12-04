import numpy as np

class L1:
    def __init__(self, alpha):
        self.alpha = alpha
    
    def call(self, weights):
        dL1_dW = self.alpha * np.sign(weights)
        return dL1_dW
    
class L2:
    def __init__(self, alpha):
        self.alpha = alpha
    
    def call(self, weights):
        dL2_dW = 2 * self.alpha * weights
        return dL2_dW




