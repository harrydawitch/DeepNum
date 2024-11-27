import numpy as np

class Tanh:
    def __init__(self):
        self.name = 'tanh'  
        self.is_output_layer = False
        self.units = None
        self._training_= True


    def forward(self, Z):
        self.inputs = Z

        numerator = np.exp(self.inputs) - np.exp(-self.inputs)
        denominator = np.exp(self.inputs) + np.exp(-self.inputs)
        tan_h = numerator / denominator

        return tan_h
    
    def backward(self, Z):
        if self.is_output_layer:
            return Z
        numerator = np.exp(Z) - np.exp(-Z)
        denominator = np.exp(Z) + np.exp(-Z)
        tan_h = numerator / denominator

        tanh_derivative = 1 - (tan_h ** 2)

        return tanh_derivative

    def call(self, da):
        derivative = self.backward(self.inputs)
        return da * derivative

class ReLU:
    def __init__(self,threshold= 0.0, max_value= None, negative_slope = 0.0):
        self.name = 'relu'  
        self.is_output_layer = False
        self.units = None
        self.threshold = threshold
        self.max_value = max_value
        self.negative_slope = negative_slope
        self._training_= True

    
    def forward(self, Z):
        self.inputs = Z

        self.inputs = np.where(self.inputs <= self.threshold, self.inputs * self.negative_slope, self.inputs)

        if self.max_value:
            return np.minimum(self.inputs, self.max_value)
        
        return self.inputs
    
    def backward(self, Z):
        if self.is_output_layer:
            return Z
        
        da = np.where(Z <= self.threshold, self.negative_slope, 1)

        if self.max_value:
            da = np.where(Z > self.max_value, 0, da)
            
        return da
    
    def call(self, da):
        derivative = self.backward(self.inputs)
        return da * derivative
    
class Sigmoid:
    def __init__(self):
        self.name = 'sigmoid'  
        self.is_output_layer = False
        self._training_= True
        self.units = None

    def forward(self, Z):
        self.inputs = Z
        sigmoid = 1 / (1 + np.exp(-self.inputs))
        return sigmoid

    def backward(self, Z):

        if self.is_output_layer:
            return Z
        
        sigmoid = 1 / (1 + np.exp(-Z))
        return sigmoid * (1 - sigmoid)

    def call(self, da):
        derivative = self.backward(self.inputs)
        return da * derivative

class Softmax:
    def __init__(self):
        self.name = 'softmax'
        self.is_output_layer = False
        self.units = None
        self._training_= True
    
    def forward(self, Z):
        self.inputs = Z

        exp = np.exp(self.inputs - np.max(Z, axis=0, keepdims=True))
        return exp / (np.sum(exp, axis=0, keepdims=True) + 1e-8)
    
    def backward(self, Z):
        if self.is_output_layer:
            return Z # I have not yet wanted to code this so i just return Z
        
        return Z

    def call(self, da):
        derivative = self.backward(self.inputs)
        return da * derivative
    
    
def activate(Z, name, derivative= False):
    activations = {'relu': ReLU,
                 'sigmoid': Sigmoid,
                 'tanh': Tanh,
                 'softmax': Softmax}
    if name is None:
        return 1

    if name in activations.keys():
        if derivative:
            return activations[name]().backward(Z)

        return activations[name]().forward(Z)
    else:
        raise NameError(f'The activation name provided is incorrect. Please check and re-enter!')