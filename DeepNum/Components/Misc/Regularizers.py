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

class ElasticNet:
    def __init__(self, alpha1, alpha2):
        if not alpha1 and not alpha2:
            raise ValueError("Missing input for alpha1 or alpha2")
        
        self.alpha1= alpha1
        self.alpha2= alpha2

    def call(self, weights):
        dElasticNet_dW = (self.alpha1 * np.sign(weights)) + (2 * self.alpha2 * weights)
        return dElasticNet_dW


