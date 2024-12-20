import numpy as np
from Components.Utils import find_shape
from Components.Layers.Base import Layer



class BatchNormalization(Layer):
    def __init__(self,  momentum= 0.99):
        super().__init__()

        self.momentum = momentum
        self.epsilon = 1e-8
        self.learnable = True
        self.parameters = {}
        self.grads = {}
        self.name = 'BatchNormalization'


    def normalize(self):

        m = self.data_in.shape[find_shape(self.data_in, mode= 'samples')]
        self.mean = np.sum(self.data_in, axis= find_shape(self.data_in, mode= 'samples'), keepdims= False) / m
        self.var =  np.sum((self.data_in - self.mean)**2, axis= find_shape(self.data_in, mode= 'samples'), keepdims= False) / m
        
        x_normalized = (self.data_in - self.mean) / np.sqrt(self.var + self.epsilon)
        return x_normalized
      



    def forward(self, inputs):

        if self.shape is None:
            self.shape = inputs.shape[find_shape(inputs, mode= 'features')]
            super().init_params()



        self.data_in = inputs
        if self._training_:
            self.x_normalized = self.normalize()

            self.running_mean = self.momentum * self.running_mean + (1-self.momentum) * self.mean
            self.running_var = self.momentum * self.running_var + (1-self.momentum) * self.var
        
        else:
            self.x_normalized = ( self.data_in - self.running_mean) / np.sqrt(self.running_var + self.epsilon)
    
        data_out = self.x_normalized * self.parameters['Weight'] + self.parameters['bias']         
        return data_out
    



    def backward(self, dout):

        m = dout.shape[find_shape(dout, mode='samples')]

        self.grads['dW'] = np.sum(dout * self.x_normalized, axis= find_shape(dout, mode='samples'), keepdims= False)
        self.grads['db'] = np.sum(dout, axis= find_shape(dout, mode='samples'), keepdims= False)

        dX_hat = dout * self.parameters['Weight']
        dvar = np.sum((dX_hat * (self.data_in - self.mean)) * -0.5 * (self.var + self.epsilon)**(-3/2), \
                      axis= find_shape(dX_hat, mode='samples'), keepdims= False)
        
        dmean = np.sum(dX_hat * -1 / np.sqrt(self.var + self.epsilon), axis= find_shape(dX_hat, mode='samples'), keepdims= False) + \
                        dvar * np.sum(-2 * (self.data_in - self.mean), axis= find_shape(dX_hat, mode='samples'), keepdims= False) / m

        dout = (dX_hat * (1/np.sqrt(self.var + self.epsilon))) + (dvar * (2*(self.data_in - self.mean) / m)) + (dmean * (1/m))

        
        return dout  