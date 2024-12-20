import numpy as np
from Components.Utils import find_shape
from Components.Layers.Base import Layer

class Tanh(Layer):
    def __init__(self):
        super().__init__()

        self.name = 'tanh'  

    def forward(self, Z):
        self.data_in = Z

        data_out = (np.exp(self.data_in) - np.exp(-self.data_in)) / (np.exp(self.data_in) + np.exp(-self.data_in))

        return data_out
    
    def backward(self, dout):

        tan_h = (np.exp(self.data_in) - np.exp(-self.data_in)) / (np.exp(self.data_in) + np.exp(-self.data_in))
        tanh_derivative = 1 - (tan_h ** 2)

        return dout * tanh_derivative
    
    def __call__(self, Z):
        return self.forward(Z)



class ReLU(Layer):
    def __init__(self,threshold= 0.0, max_value= None, negative_slope = 0.0):
        super().__init__()

        self.name = 'relu'  
        self.threshold = threshold
        self.max_value = max_value
        self.negative_slope = negative_slope

    
    def forward(self, inputs):
        self.data_in = inputs

        data_out = np.where(self.data_in <= self.threshold, self.data_in * self.negative_slope, self.data_in)

        if self.max_value:
            data_out = np.minimum(data_out, self.max_value)
        
        return data_out
    

    def backward(self, dout):
        
        derivative = np.where(self.data_in <= self.threshold, self.negative_slope, 1)

        if self.max_value:
            derivative[self.data_in > self.max_value] = 0
            
        return dout * derivative
    
    def __call__(self, Z):
        return self.forward(Z)
    
    
    
class Sigmoid(Layer):
    def __init__(self):
        super().__init__()

        self.name = 'sigmoid'  

    def forward(self, Z):
        self.data_in = Z
        data_out = 1 / (1 + np.exp(-self.data_in))
        return data_out

    def backward(self, dout):
        
        sigmoid = 1 / (1 + np.exp(-self.data_in))
        return dout * (sigmoid * (1 - sigmoid))

    def __call__(self, Z):
        return self.forward(Z)


class Softmax(Layer):
    def __init__(self):
        super().__init__()

        self.name = 'softmax'
    
    def forward(self, Z):
         
        exp = np.exp(Z - np.max(Z, axis=find_shape(Z, mode='features'), keepdims=True))
        data_out = exp / (np.sum(exp, axis=find_shape(exp, mode='features'), keepdims=True) + 1e-8)
        return data_out
    
    def backward(self, dout):
        return dout
        

    def __call__(self, Z):
        return self.forward(Z)


    
    

