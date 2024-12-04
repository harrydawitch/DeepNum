import numpy as np
from .utils import *  
import copy

class Container:
    def __init__(self):
        self.layers = []
        self.losses =  None
        self.optimizer = None
        self.metrics = None
        self.transformational_Layers = ["Dense", "Conv2d"]
        self.support_layers = ["Dropout, BatchNormalization"]
        self.learnable_layers = ['Dense', 'Conv2d', 'BatchNormalization']
        self.evaluation = False
        self.parameters = {}

    def setup_layers(self):
        units = self.layers[0].units
        if self.layers[0].input_size is None:
            raise ValueError("Please assign input_size for the first layer")

        for i in range(1, len(self.layers)):

            if self.layers[i].name in self.learnable_layers:
                self.layers[i].input_size = units
                self.layers[i].init_params()
                units = self.layers[i].units
            
            else:
                continue


    def forward_propagation(self, X):
        inputs = X

        for layer in self.layers:
            inputs = layer.forward(inputs)

        return inputs

    def backward_propagation(self, y):


        dout = pick_loss(name= self.losses, y_pred= self.layers[-1].outputs, y_true= y, derivative= True)

        for layer in reversed(self.layers):
            dout = layer.backward(dout= dout)

            if layer.name in self.learnable_layers:
                update_parameter(optimizer= self.optimizer, layer= layer)


class Model(Container):
    def __init__(self):
        super().__init__()

    def add(self, layer):
        self.layers.append(layer)


    def fit(self, X, y, batch_size= None, epochs= None, verbose= False, show_metric= False, patience= float('inf')):
        m = X.shape[1]
        if not batch_size:
            batch_size = m
        
        if not epochs:
            raise KeyError('Epoch is not specify')
        
        

        super().setup_layers()


        previous_loss = float('inf')
        permutation = np.random.permutation(m)
        patience_counter = 0
        history = {'loss': [],
                   'accuracy': []}


        for epoch in range(1, epochs + 1):
            X_shuffled = X[:, permutation]
            y_shuffled = y[:, permutation]
            total_batches_loss = 0 

            for batch in range(0, m, batch_size):
                X_batches = X_shuffled[:, batch:batch + batch_size] 
                y_batches = y_shuffled[:, batch:batch + batch_size]

                batch_samples = X_batches.shape[1]

                y_pred = super().forward_propagation(X_batches)
                super().backward_propagation(y_batches)

                batch_loss = pick_loss(name= self.losses, y_pred= y_pred, y_true= y_batches, derivative= False)
                total_batches_loss += (batch_loss * batch_samples)
            
            epoch_loss = total_batches_loss / m
            history['loss'].append(epoch_loss)

            if verbose:
                print(f'Epoch {epoch}/{epochs} - Loss: {epoch_loss}')

            if show_metric:
                prediction = self.forward_propagation(X)
                metric_value = pick_metric(metrics= self.metrics, y_pred= prediction, y_true= y)
                history['accuracy'].append(metric_value)
                print(f'{self.metrics.capitalize()}: {metric_value * 100:.2f}%')

            if previous_loss > epoch_loss:
                previous_loss = epoch_loss
                self.best_parameters = copy.deepcopy(self.layers)
                patience_counter = 0

            else:
                patience_counter += 1  # Increment patience counter
                if patience < patience_counter:  # Check for early stopping condition
                    print(f'Early Stopping at epoch: {epoch}')
                    break
        return history




    def compile(self, optimizer= None, losses= None, metrics= None):
        self.optimizer = optimizer
        self.losses = losses    
        self.metrics = metrics



    def predict(self, X):
        inputs = X

        for layer in self.best_parameters:
            if layer.name in self.layers_type:
                inputs = layer.forward(inputs)
            elif layer.name in self.activations_type:
                inputs = layer.forward(inputs)
        
        # inputs = np.argmax(inputs, axis= 0)
        return inputs

    def save_model(self, filepath):
        np.savez(filepath, 
                layers=self.best_parameters,
                layers_type= self.layers_type,
                activations_type= self.activations_type)


    def load_model(self, filepath):
        data = np.load(filepath, allow_pickle=True)
        
        self.layers = list(data['layers'])
        self.activation_names = list(data['activation_names'])


