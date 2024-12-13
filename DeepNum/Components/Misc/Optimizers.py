import numpy as np
import copy
from Components.Utilities.Utils import find_shape

class Optimizer():
    def __init__(self):
        self.contain_params = False




class Adam(Optimizer):
    def __init__(self, learning_rate= 0.001, beta1= 0.9, beta2= 0.999, epsilon= 1e-8, clip = float('inf'), *args):
        super().__init__(*args)


        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.clip = clip
        self.name = 'adam'
        self.time = 0
        


    def call(self, layer):

        self.time += 1

        if not self.contain_params:
            self.v_dw = np.zeros((layer.parameters['Weight'].shape))
            self.v_db = np.zeros((layer.parameters['bias'].shape))
            self.s_dw = np.zeros((layer.parameters['Weight'].shape))
            self.s_db = np.zeros((layer.parameters['bias'].shape))

            self.contain_params = True
        
        self.v_dw = self.beta1 * self.v_dw + (1 - self.beta1) * layer.grads['dW']
        self.v_db = self.beta1 * self.v_db + (1 - self.beta1) * layer.grads['db']
        self.s_dw = self.beta2 * self.s_dw + (1 - self.beta2) * (layer.grads['dW'] ** 2)
        self.s_db = self.beta2 * self.s_db + (1 - self.beta2) * (layer.grads['db'] ** 2)

        v_dw_cb = self.v_dw / (1 - (self.beta1 ** self.time))
        v_db_cb = self.v_db / (1 - (self.beta1 ** self.time))
        s_dw_cb = self.s_dw / (1 - (self.beta2 ** self.time))
        s_db_cb = self.s_db / (1 - (self.beta2 ** self.time))

        # Compute and clip updates
        weight_update = v_dw_cb / (np.sqrt(s_dw_cb) + self.epsilon)
        weight_update = np.clip(weight_update, -self.clip, self.clip)
        bias_update = v_db_cb / (np.sqrt(s_db_cb) + self.epsilon)
        bias_update = np.clip(bias_update, -self.clip, self.clip)

        layer.parameters['Weight'] -= self.learning_rate * weight_update
        layer.parameters['bias'] -= self.learning_rate * bias_update

#===========================================================================================================================================================

class RMSprop(Optimizer):
    def __init__(self, learning_rate= 0.001, beta= 0.9, epsilon= 1e-7, clip= float('inf'), *args):
        super().__init__(*args)


        self.learning_rate = learning_rate
        self.beta = beta
        self.epsilon = epsilon
        self.clip = clip
        self.name = 'rmsprop'
        self.main_layers = ['Dense']




    def call(self, layer):

        if not self.contain_params:
            self.s_dw = np.zeros((layer.parameters['Weight'].shape))
            self.s_db = np.zeros((layer.parameters['bias']))

            self.contain_params = True

        self.s_dw = self.beta * self.s_dw + (1 - self.beta) * (layer.grads['dW'] ** 2)
        self.s_db = self.beta * self.s_db + (1 - self.beta) * (layer.grads['db'] ** 2)

        # Compute and clip updates
        weight_update = layer.grads['dW'] / (np.sqrt(self.s_dw) + self.epsilon)
        weight_update = np.clip(weight_update, -self.clip, self.clip)
        bias_update = layer.grads['db'] / (np.sqrt(self.s_db) + self.epsilon)
        bias_update = np.clip(bias_update, -self.clip, self.clip)

        layer.parameters['Weight'] -= self.learning_rate * weight_update
        layer.bias -= self.learning_rate * bias_update

#===========================================================================================================================================================

class Momentum(Optimizer):
    def __init__(self, learning_rate= 0.001, beta= 0.9, clip= float('inf'), *args):
        super().__init__(*args)


        self.learning_rate = learning_rate
        self.beta = beta
        self.clip = clip
        self.name = 'momentum'



    def call(self, layer):

        if not self.contain_params:
            self.v_dw = np.zeros((layer.parameters['Weight'].shape))
            self.v_db = np.zeros((layer.parameters['bias'].shape))
            self.s_dw = np.zeros((layer.parameters['Weight'].shape))
            self.s_db = np.zeros((layer.parameters['bias'].shape))

            self.contain_params = True


        self.v_dw = self.beta * self.v_dw + (1 - self.beta) * layer.grads['dW']
        self.v_db = self.beta * self.v_db + (1 - self.beta) * layer.grads['db']

        # Compute and clip updates
        weight_update = np.clip(self.v_dw, -self.clip, self.clip)
        bias_update = np.clip(self.v_db, -self.clip, self.clip)

        layer.parameters['Weight'] -= self.learning_rate * weight_update
        layer.parameters['bias'] -= self.learning_rate * bias_update

#===========================================================================================================================================================

class Gradient_Descent(Optimizer):
    def __init__(self, learning_rate=0.001, clip=float('inf'), *args):
        super().__init__(*args)


        self.learning_rate = learning_rate
        self.clip = clip
        self.name = 'sgd'



    def call(self, layer):

        # Compute and clip updates
        layer.grads['dW'] = np.clip(layer.grads['dW'], -self.clip, self.clip)
        layer.grads['db'] = np.clip(layer.grads['db'], -self.clip, self.clip)

        # Update parameters
        layer.parameters['Weight'] -= self.learning_rate * layer.grads['dW']
        layer.parameters['bias'] -= self.learning_rate * layer.grads['db']



    
