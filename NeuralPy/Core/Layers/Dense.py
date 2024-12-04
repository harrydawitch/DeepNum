import numpy as np
from ..utils import *

class Dense:
    def __init__(self, units, input_size = None, activation= None, initializer= None, regularizer= None):
        self.units= units
        self.input_size= input_size
        self.initializer= initializer
        self.regularizer = regularizer
        self.activation = activation
        self.optimizer = None
        self.parameters = {}
        self._training_= True
        self.name = 'Dense'

        if activation is not None:
            self.activation = activate(name= activation)

        if self.input_size is not None:
            self.init_params()



    def init_params(self):
        if self.initializer:
            self.weight = pick_initializer(initializers= self.initializer, shape= (self.units, self.input_size))
        else:
            self.weight = np.random.randn(self.units, self.input_size)

        self.bias = np.zeros((self.units, 1))
        



    def forward(self, inputs):

        self.inputs = inputs

        Z = np.matmul(self.weight, self.inputs) + self.bias 
        self.Z = Z

        if self.activation is not None:
            self.outputs= self.activation.forward(Z= self.Z)
        else:   
            self.outputs = Z
        
        return self.outputs


    def backward(self, dout): 
        m = dout.shape[1]
        

        if self.activation is not None:
            dZ = self.activation.backward(dout= dout)
        else:
            dZ = dout


        dW = np.matmul(dZ, self.inputs.T) / m
        db = np.sum(dZ, axis= 1, keepdims= True) / m


        if self.regularizer is not None:
            penalty= self.regularizer.call(self.weight)
            dW += penalty


        self.dw = dW
        self.db = db

        prev_dout = np.matmul(self.weight.T, dZ)

        return prev_dout
        

