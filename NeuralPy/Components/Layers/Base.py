import numpy as np
from ..Utils import activate, find_shape, pick_initializer

class Container:
    def __init__(self):
        self.input = None
        self.layers = []
        self.losses =  None
        self.optimizer = None
        self.metrics = None
        self.transformational_Layers = ["Dense", "Conv2D"]
        self.support_layers = ["Dropout, BatchNormalization"]
        self.learnable_layers = ['Dense', 'Conv2D', 'BatchNormalization']
        self.evaluation = False
        self.parameters = {}

    def setup_layers(self):
        input = self.input['X_train'].shape[find_shape(self.layers[0], mode='features')]
        
        
        for i in range(0, len(self.layers)):

            if self.layers[i].name in self.learnable_layers:
                self.layers[i].input = input
                self.layers[i].init_params()
                input = self.layers[i].output
            
            else:
                continue



class Layer(Container):
    def __init__(self):
        self.input = None
        self.layers = []
        self.connect = self.input['X_train'].shape[find_shape(self.input['X_train'], mode='features')]
    
    def setup_layers(self, layer):
        if layer.learnable:
            layer.input = self.connect
            layer.init_params(layer)

            self.connect - layer.output


            if hasattr(layer, 'activation') and layer.activation is not None:
                activation_layer = activate(layer.activation)
            
    def init_params(self, layer):

        if layer.name == 'Dense':
            layer.weight = pick_initializer(initializers= layer.initializer, 
                                            shape= (layer.output, layer.input), 
                                            name= layer.name)    
            layer.bias = np.zeros((layer.output, 1))

        

        elif layer.name == 'Conv2D':
            
            assert len(self.filter_size) == 2, "Filter size must be a tuple/list contain 2 values -> (Height, Width)"
            height, width = self.filter_size
            layer.weight = pick_initializer(initializers= layer.initializer, 
                                            shape= (height, width, layer.input, layer.output), 
                                            name= layer.name)
            layer.bias = np.zeros(layer.output)

                
            layer.filter_height = height
            layer.filter_width= width
        


        elif layer.name == 'BatchNormalization':
            
            self.weight = np.ones((self.input, 1)) 
            self.bias = np.zeros((self.input, 1))

            self.running_mean = np.zeros((self.input, 1))
            self.running_var = np.ones((self.input, 1))
            
            self.output = self.input
