import numpy as np

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


    def forward_propagation(self, X):
        inputs = X

        for layer in self.layers:
            inputs = layer.forward(inputs)

        return inputs
    

    def backward_propagation(self, y):


        dout = pick_loss(name= self.losses, y_pred= self.layers[-1].data_out, y_true= y, derivative= True)

        for layer in reversed(self.layers):
            dout = layer.backward(dout= dout)

            if layer.name in self.learnable_layers:
                update_parameter(optimizer= self.optimizer, layer= layer)