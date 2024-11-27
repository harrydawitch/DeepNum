import numpy as np
from Activations.Activations import *
from Initializers.Initializers import *

class Dense:
    def __init__(self, units, input_size = None, activation= None, initializer= None, regularizer= None):
        self.units= units
        self.input_size= input_size
        self.activation = activation
        self.initializer= initializer
        self.regularizer = regularizer
        self.is_output_layer = False
        self.learnable = True
        self._training_= True
        self.name = 'Dense'

        if self.input_size is not None:
            self._init_params()



    def _init_params(self):
        if self.initializer:
            self.weight = initializer(initializers= self.initializer, shape= (self.units, self.input_size))
        else:
            self.weight = np.random.randn(self.units, self.input_size)

        self.bias = np.zeros((self.units, 1))




    def forward(self, inputs):

        self.inputs = inputs

        Z = np.matmul(self.weight, self.inputs) + self.bias 
        self.Z = Z

        if self.activation is not None:
            self.outputs= activate(Z= Z, name= self.activation, derivative= False)
        else:   
            self.outputs = Z
        
        return self.outputs


    def backward(self, da): 
        m = da.shape[1]
        
        if self.is_output_layer:
            dZ = da
        else:
            dZ = da * activate(Z= self.Z, name= self.activation, derivative= True)
        dW = np.matmul(dZ, self.inputs.T) / m
        db = np.sum(dZ, axis= 1, keepdims= True) / m

        self.dw = dW
        self.db = db

        prev_da = np.matmul(self.weight.T, dZ)

        return prev_da
        

