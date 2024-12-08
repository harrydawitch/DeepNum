import numpy as np
from ..Utils import find_shape

class Tanh:
    def __init__(self):
        self.name = 'tanh'  


    def forward(self, Z):
        self.inputs = Z

        numerator = np.exp(self.inputs) - np.exp(-self.inputs)
        denominator = np.exp(self.inputs) + np.exp(-self.inputs)
        self.outputs = numerator / denominator

        return self.outputs
    
    def backward(self, dout):

        numerator = np.exp(self.inputs) - np.exp(-self.inputs)
        denominator = np.exp(self.inputs) + np.exp(-self.inputs)
        tan_h = numerator / denominator

        tanh_derivative = 1 - (tan_h ** 2)

        return dout * tanh_derivative
    
    def __call__(self, Z):
        return self.forward(Z)


class ReLU:
    def __init__(self,threshold= 0.0, max_value= None, negative_slope = 0.0):
        self.name = 'relu'  
        self.is_output_layer = False
        self.threshold = threshold
        self.max_value = max_value
        self.negative_slope = negative_slope

    
    def forward(self, Z):
        self.inputs = Z

        self.outputs = np.where(self.inputs <= self.threshold, self.inputs * self.negative_slope, self.inputs)

        if self.max_value:
            self.outputs = np.minimum(self.outputs, self.max_value)
        
        return self.outputs
    

    def backward(self, dout):
        
        derivative = np.where(self.inputs <= self.threshold, self.negative_slope, 1)

        if self.max_value:
            derivative[self.inputs > self.max_value] = 0
            
        return dout * derivative
    
    def __call__(self, Z):
        return self.forward(Z)
    
    
    
class Sigmoid:
    def __init__(self):
        self.name = 'sigmoid'  
        self.is_output_layer = False

    def forward(self, Z):
        self.inputs = Z
        self.outputs = 1 / (1 + np.exp(-self.inputs))
        return self.outputs

    def backward(self, dout):
        
        sigmoid = 1 / (1 + np.exp(-self.inputs))
        derivative=  sigmoid * (1 - sigmoid)
        return dout * derivative

    def __call__(self, Z):
        return self.forward(Z)


class Softmax:
    def __init__(self, custom_mode= False):
        self.name = 'softmax'
        self.is_output_layer = False
        self.custom_mode= custom_mode
    
    def forward(self, Z):
        self.inputs = Z
        exp = np.exp(self.inputs - np.max(Z, axis=find_shape(Z, mode='features'), keepdims=True))
        self.outputs = exp / (np.sum(exp, axis=find_shape(Z, mode='features'), keepdims=True) + 1e-8)
        return self.outputs
    
    def backward(self, dout):
        S = self.outputs  # Shape: (n_classes, batch_size)
        m = dout.shape[1]
        
        if self.custom_mode:
            # Compute the Jacobian of softmax
            jacobian = np.einsum('ij,kj->ikj', S, S)  # Shape: (n_classes, n_classes, batch_size)
            jacobian -= np.eye(S.shape[0])[:, :, np.newaxis]  # Subtract identity for the diagonal
            
            # Multiply with the dout (gradient of loss wrt softmax outputs)
            dZ = np.sum(jacobian * dout[np.newaxis, :, :], axis=1)  # Sum over the second axis (classes)
            
            return dZ
        
        else:
            return dout
        

    def __call__(self, Z):
        return self.forward(Z)


    
    

