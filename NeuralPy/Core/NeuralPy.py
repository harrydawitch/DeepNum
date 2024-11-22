import numpy as np
from Optimizers.Optimizers import *
from Metrics.Metrics import *
from Losses.Losses import *
from Optimizers.Optimizers import *
import copy

class NeuralPy:
    def __init__(self):
        self.layers = []
        self.layers_type = ['Dense', 'Dropout']
        self.activations_type = ['relu', 'tanh', 'sigmoid', 'softmax']

    def add(self, layer):

        if layer.name in self.layers_type:
            if layer.name == 'Dense':
                if layer.input_size is None:
                    layer.input_size = self.layers[-1].units
                    layer._init_params()

            elif layer.name == 'Dropout':
                layer.units = self.layers[-1].units
        
        elif layer.name in self.activations_type:
            layer.units = self.layers[-1].units
            
        # Append current layer to layers list attribute    
        self.layers.append(layer)

    def fit(self, X, y, batch_size= None, epochs= None, verbose= False, patience= float('inf')):
        m = X.shape[1]
        if not batch_size:
            batch_size = m
        
        if not epochs:
            raise KeyError('Epoch is not specify')


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

                y_pred = self._forward_propagation_(X_batches)
                self._backward_propagation_(y_batches)

                batch_loss = pick_loss(name= self.losses, y_pred= y_pred, y_true= y_batches, derivative= False)
                total_batches_loss += (batch_loss * batch_samples)
            
            prediction = self._forward_propagation_(X)
            metric_value = pick_metric(metrics= self.metrics, y_pred= prediction, y_true= y)
            epoch_loss = total_batches_loss / m
            history['loss'].append(epoch_loss)
            history['accuracy'].append(metric_value)
        
            if verbose:
                print(f'Epoch {epoch}/{epochs} - Loss: {epoch_loss} - {self.metrics.capitalize()}: {metric_value * 100:.2f}%')

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
        
        inputs = np.argmax(inputs, axis= 0)
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

    def _forward_propagation_(self, X):
        inputs = X

        for layer in self.layers:

            if layer.name in self.layers_type:
                inputs = layer.forward(inputs)
            elif layer.name in self.activations_type:
                inputs = layer.forward(inputs)
        return inputs

    def _backward_propagation_(self,  y):

        self.layers[-1].is_output_layer = True

        da = pick_loss(name= self.losses, y_pred= self.layers[-1].outputs, y_true= y, derivative= True)

        for layer in reversed(self.layers):
            if layer.name in self.layers_type:
                da = layer.backward(da ,y)

                if layer.learnable:
                    update_parameter(optimizer= self.optimizer, layer= layer)

            elif layer.name in self.activations_type:
                da = layer.call(da= da)

