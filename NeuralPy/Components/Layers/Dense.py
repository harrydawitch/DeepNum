import numpy as np
from ..Utils import pick_initializer, find_shape

class Dense:
    def __init__(self, units, activation= None, initializer= None, regularizer= None):
        self.input= None
        self.output= units
        self.initializer= initializer
        self.regularizer = regularizer
        self.activation = activation
        self.optimizer = None
        self.parameters = {}
        self._training_= True
        self.learnable = True
        self.name = 'Dense'




    def init_params(self):
        if self.initializer:
            self.weight = pick_initializer(initializers= self.initializer, 
                                           shape= (self.output, self.input), 
                                           name= self.name)
            
        else:
            self.weight = np.random.randn(self.output, self.input)

        self.bias = np.zeros((self.output, 1))
        
        
        
    def forward(self, inputs):

        self.data_in = inputs

        Z = np.matmul(self.weight, self.data_in) + self.bias 
        self.Z = Z

        if self.activation is not None:
            self.data_out= self.activation.forward(Z= self.Z)
        else:   
            self.data_out = Z
        
        return self.data_out
    
    
    
    def backward(self, dout): 
        m = dout.shape[find_shape(dout, mode='samples')]
        

        if self.activation is not None:
            dZ = self.activation.backward(dout= dout)
        else:
            dZ = dout


        dW = np.matmul(dZ, self.data_in.T) / m
        db = np.sum(dZ, axis= find_shape(dZ, mode='samples'), keepdims= True) / m


        if self.regularizer is not None:
            penalty= self.regularizer.call(self.weight)
            dW += penalty


        self.dw = dW
        self.db = db

        prev_dout = np.matmul(self.weight.T, dZ)

        return prev_dout