import numpy as np
from Components.Utils import find_shape
from Components.Layers.Base import Layer


class Dense(Layer):
    def __init__(self, units, initializer= 'glorot_uniform', regularizer= None):
        super().__init__()
        self.output= units
        self.initializer= initializer
        self.regularizer = regularizer
        self.learnable= True
        self.parameters = {}
        self.grads = {}
        self.name = 'Dense'


        
        
    def forward(self, inputs):

        if self.shape is None:
            self.shape = inputs.shape[find_shape(inputs, mode= 'features')]
            super().init_params()



        self.data_in = inputs

        Z = np.matmul(self.data_in, self.parameters['Weight']) + self.parameters['bias']
        
        return Z
    
    
    
    def backward(self, dout): 
        m = dout.shape[find_shape(dout, mode='samples')]
        
        self.grads['dW'] = np.matmul(self.data_in.T, dout) / m
        self.grads['db'] = np.sum(dout, axis= find_shape(dout, mode='samples'), keepdims= True) / m


        if self.regularizer is not None:
            penalty= self.regularizer.call(self.parameters['Weight'])
            self.grads['dW'] += penalty


        dout = np.matmul(dout, self.parameters['Weight'].T)

        return dout
