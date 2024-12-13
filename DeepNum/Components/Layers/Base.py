import numpy as np
from Components.Utilities.Utils import pick_initializer

class Layer():
    def __init__(self):
        self.shape = None
        self.output = None
        self.optimizer = None
        self._training_= True
        self.learnable = False

    def forward(self, inputs):
        raise NotImplementedError
    
    def backward(self, dout):
        raise NotImplementedError
        


    def init_params(self):

        if self.name == 'Dense':
            self.parameters['Weight'] = pick_initializer(initializers= self.initializer, 
                                            shape= (self.output, self.shape), 
                                            name= self.name)    
            self.parameters['bias'] = np.zeros((self.output, 1))

        
        elif self.name == 'Conv2D':
            

            self.parameters['Weight'] = pick_initializer(initializers= self.initializer, 
                                            shape= (self.output, self.shape, self.filter_size, self.filter_size), 
                                            name= self.name)
            
            self.parameters['bias'] = np.zeros(self.output)

    


        elif self.name == 'BatchNormalization':

            self.parameters['Weight'] = np.ones((self.shape, 1)) 
            self.parameters['bias'] = np.zeros((self.shape, 1))

            self.running_mean = np.zeros((self.shape, 1))
            self.running_var = np.ones((self.shape, 1))
            
            self.output = self.shape 



        
