import numpy as np
import copy

class Adam:
    def __init__(self, learning_rate= 0.001, beta1= 0.9, beta2= 0.999, epsilon= 1e-8, clip = float('inf')):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.clip = clip
        self.name = 'adam'
        self._contain_parameters = False
        self.main_layers = ['Dense']
        self.t = 0

    def call(self, layer):
        n, m = layer.weight.shape
        self.t += 1
        
        if not self._contain_parameters:
            self.v_dw = np.zeros((n, m))
            self.v_db = np.zeros((n, 1))
            self.s_dw = np.zeros((n, m))
            self.s_db = np.zeros((n, 1))

            self._contain_parameters = True
        
        
        self.v_dw = self.beta1 * self.v_dw + (1 - self.beta1) * layer.dw
        self.v_db = self.beta1 * self.v_db + (1 - self.beta1) * layer.db
        self.s_dw = self.beta2 * self.s_dw + (1 - self.beta2) * (layer.dw ** 2)
        self.s_db = self.beta2 * self.s_db + (1 - self.beta2) * (layer.db ** 2)

        self.v_dw_cb = self.v_dw / (1 - (self.beta1 ** self.t))
        self.v_db_cb = self.v_db / (1 - (self.beta1 ** self.t))
        self.s_dw_cb = self.s_dw / (1 - (self.beta2 ** self.t))
        self.s_db_cb = self.s_db / (1 - (self.beta2 ** self.t))

        # Compute and clip updates
        weight_update = self.v_dw_cb / (np.sqrt(self.s_dw_cb) + self.epsilon)
        weight_update = np.clip(weight_update, -self.clip, self.clip)
        bias_update = self.v_db_cb / (np.sqrt(self.s_db_cb) + self.epsilon)
        bias_update = np.clip(bias_update, -self.clip, self.clip)

        layer.weight -= self.learning_rate * weight_update
        layer.bias -= self.learning_rate * bias_update

class RMSprop:
    def __init__(self, learning_rate= 0.001, beta= 0.9, epsilon= 1e-7, clip= float('inf')):
        self.learning_rate = learning_rate
        self.beta = beta
        self.epsilon = epsilon
        self.clip = clip
        self.name = 'rmsprop'
        self.main_layers = ['Dense']
        self.contain_parameters = False

    def call(self, layer):
        n, m = layer.weight.shape

        if not self.contain_parameters:
            self.s_dw = np.zeros((n, m))
            self.s_db = np.zeros((n, 1))
            
            self.contain_parameters = True


        self.s_dw = self.beta * self.s_dw + (1 - self.beta) * (layer.dw ** 2)
        self.s_db = self.beta * self.s_db + (1 - self.beta) * (layer.db ** 2)

        # Compute and clip updates
        weight_update = layer.dw / (np.sqrt(self.s_dw) + self.epsilon)
        weight_update = np.clip(weight_update, -self.clip, self.clip)
        bias_update = layer.db / (np.sqrt(self.s_db) + self.epsilon)
        bias_update = np.clip(bias_update, -self.clip, self.clip)

        layer.weight -= self.learning_rate * weight_update
        layer.bias -= self.learning_rate * bias_update

class Momentum:
    def __init__(self, learning_rate= 0.001, beta= 0.9, clip= float('inf')):
        self.learning_rate = learning_rate
        self.beta = beta
        self.clip = clip
        self.name = 'momentum'
        self.main_layers = ['Dense']
        self.contain_parameters= False

    def call(self, layer):
        n, m = layer.weight.shape

        if not self.contain_parameters:
            self.v_dw = np.zeros((n, m))
            self.v_db = np.zeros((n, 1))
        
            self.contain_parameters = True

        self.v_dw = self.beta * self.v_dw + (1 - self.beta) * layer.dw
        self.v_db = self.beta * self.v_db + (1 - self.beta) * layer.db

        # Compute and clip updates
        weight_update = np.clip(self.v_dw, -self.clip, self.clip)
        bias_update = np.clip(self.v_db, -self.clip, self.clip)

        layer.weight -= self.learning_rate * weight_update
        layer.bias -= self.learning_rate * bias_update

class Gradient_Descent:
    def __init__(self, learning_rate=0.001, clip=float('inf')):
        self.learning_rate = learning_rate
        self.clip = clip
        self.name = 'sgd'
        self.main_layers = ['Dense']


    def call(self, layer):

        # Compute and clip updates
        layer.dw = np.clip(layer.dw, -self.clip, self.clip)
        layer.db = np.clip(layer.db, -self.clip, self.clip)

        # Update parameters
        layer.weight -= self.learning_rate * layer.dw
        layer.bias -= self.learning_rate * layer.db


def update_parameter(optimizer, layer):
    optimizer_classes = (Adam, RMSprop, Momentum, Gradient_Descent)

    if isinstance(optimizer, optimizer_classes):
        clone = copy.deepcopy(optimizer)
        clone.call(layer= layer)
    elif isinstance(optimizer, str):
        pick_optimizer(name= optimizer, layer= layer)
    else:
        raise ValueError('You are enter wrong optimizer object type. Please check again!')


def pick_optimizer(name, layer, **kwargs):
    optimizer = {'adam': Adam,
                 'sgd': Gradient_Descent,
                 'momentum': Momentum,
                 'rmsprop': RMSprop}
    
    if name in optimizer.keys():
        return optimizer[name]().call(layer= layer)
    else:
        raise NameError(f'The optimizer name provided is incorrect. Please check and re-enter!')

    
