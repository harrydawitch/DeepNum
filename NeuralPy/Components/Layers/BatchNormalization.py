import numpy as np
from ..Utils import find_shape


class BatchNormalization:
    def __init__(self, input= None,  momentum= 0.99):
        self.input = None
        self.output = None
        self.momentum = momentum
        self.learnable = True
        self.optimizer = None
        self.epsilon = 1e-8
        self.parameters = {}
        self._training_= True
        self.learnable = True
        self.name = 'BatchNormalization'



    def init_params(self):
        self.weight = np.ones((self.input, 1)) 
        self.bias = np.zeros((self.input, 1))

        self.running_mean = np.zeros((self.input, 1))
        self.running_var = np.ones((self.input, 1))
        
        self.output = self.input
        



    def normalize(self, X):

        m = X.shape[find_shape(X, mode= 'samples')]
        self.mean = np.sum(X, axis= find_shape(X, mode= 'samples'), keepdims= True) / m
        self.var =  np.sum((X - self.mean)**2, axis= find_shape(X, mode= 'samples'), keepdims= True) / m
        
        x_normalized = (X - self.mean) / np.sqrt(self.var + self.epsilon)
        return x_normalized
      



    def forward(self, X):
        self.inputs = X
        if self._training_:
            self.x_normalized = self.normalize(X)

            self.running_mean = self.momentum * self.running_mean + (1-self.momentum) * self.mean
            self.running_var = self.momentum * self.running_var + (1-self.momentum) * self.var
        
        else:
            self.x_normalized = (X - self.running_mean) / np.sqrt(self.running_var + self.epsilon)
    
        outputs = self.weight * self.x_normalized + self.bias         
        return outputs
    



    def backward(self, dout):

        m = dout.shape[find_shape(dout, mode='samples')]

        self.dw = np.sum(dout * self.x_normalized, axis= find_shape(dout, mode='samples'), keepdims= True)
        self.db = np.sum(dout, axis= find_shape(dout, mode='samples'), keepdims= True)

        dX_hat = dout * self.weight
        dvar = np.sum((dX_hat * (self.inputs - self.mean)) * -0.5 * (self.var + self.epsilon)**(-3/2), \
                      axis= find_shape(dX_hat, mode='samples'), keepdims= True)
        
        dmean = np.sum(dX_hat * -1 / np.sqrt(self.var + self.epsilon), axis= find_shape(dX_hat, mode='samples'), keepdims= True) + \
                        dvar * np.sum(-2 * (self.inputs - self.mean), axis= find_shape(dX_hat, mode='samples'), keepdims= True) / m

        dout = (dX_hat * (1/np.sqrt(self.var + self.epsilon))) + (dvar * (2*(self.inputs - self.mean) / m)) + (dmean * (1/m))

        return dout  