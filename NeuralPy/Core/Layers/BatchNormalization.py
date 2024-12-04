import numpy as np

class BatchNormalization:
    def __init__(self, input_size= None,  momentum= 0.99):
        self.name = 'BatchNormalization'
        self.input_size = input_size
        self.momentum = momentum
        self.learnable = True
        self.optimizer = None
        self.epsilon = 1e-8
        self.parameters = {}
        self._training_= True

        if self.input_size:
            self._init_params()

    def init_params(self):
        self.weight = np.ones((self.input_size, 1)) 
        self.bias = np.zeros((self.input_size, 1))

        self.running_mean = np.zeros((self.input_size, 1))
        self.running_var = np.ones((self.input_size, 1))
        
        self.units = self.input_size
        
    def normalize(self, X):

        m = X.shape[1]
        self.mean = np.sum(X, axis=1, keepdims= True) / m
        self.var =  np.sum((X - self.mean)**2, axis= 1, keepdims= True) / m
        
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

        m = dout.shape[1]
        self.dw = np.sum(dout * self.x_normalized, axis= 1, keepdims= True)
        self.db = np.sum(dout, axis= 1, keepdims= True)

        dX_hat = dout * self.weight
        dvar = np.sum((dX_hat * (self.inputs - self.mean)) * -0.5 * (self.var + self.epsilon)**(-3/2), axis= 1, keepdims= True)
        dmean = np.sum(dX_hat * -1 / np.sqrt(self.var + self.epsilon), axis= 1, keepdims= True) + \
                        dvar * np.sum(-2 * (self.inputs - self.mean), axis= 1, keepdims= True) / m

        dout = (dX_hat * (1/np.sqrt(self.var + self.epsilon))) + (dvar * (2*(self.inputs - self.mean) / m)) + (dmean * (1/m))

        return dout      


