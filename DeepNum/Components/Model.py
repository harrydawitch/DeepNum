    
import numpy as np
import copy
from Components.Utilities.Utils import find_shape, pick_loss


class Sequential:
    def __init__(self, X_train= None, y_train= None):
        if X_train is None or y_train is None:
            raise ValueError(f'Missing input! neither X_train nor y_train')
        
        self.data = {'X_train': X_train,
                     'y_train': y_train}
        self.layers = []
        self.configuration = {}
        self.evaluation = False
        self.parameters = {}
        

    def add(self, layer):
        self.layers.append(layer)


    def config(self, optimizer= None, loss= None, metric= None):
        self.configuration['optimizer'] = optimizer
        self.configuration['loss'] = loss
        self.configuration['metric']= metric

        for layer in self.layers:
            if hasattr(layer, 'parameters'):
                layer.optimizer = copy.deepcopy(self.configuration['optimizer'])



    def forward_propagation(self, X):
        inputs = X

        for layer in self.layers:
            inputs = layer.forward(inputs)
        
        return inputs

    
    def backward_propagation(self, inputs):
        dout = inputs

        for layer in reversed(self.layers):
            dout = layer.backward(dout)


    def update_parameters(self):
        for layer in self.layers:

            if hasattr(layer, 'parameters'):
                layer.optimizer.call(layer= layer)           

        



    def train(self, batch_size=None, epochs=None, verbose=False, patience=float('inf')):
        m = self.data['X_train'].shape[find_shape(self.data['X_train'], mode='samples')]
        batch_size = batch_size or min(32, m)  # Default to 32 or full dataset size if smaller

        if not epochs or not isinstance(epochs, int) or epochs <= 0:
            raise ValueError('Invalid value for epochs. Must be a positive integer.')

        previous_loss = float('inf')
        patience_counter = 0
        history = {'loss': [], 'accuracy': []}

        for epoch in range(1, epochs + 1):
            # Shuffle data
            permutation = np.random.permutation(m)
            X_train = self.data['X_train']
            y_train = self.data['y_train']
            
            X_shuffled = X_train[:, permutation] if len(X_train.shape) <= 2 else X_train[permutation, :]
            y_shuffled = y_train[:, permutation]

            for batch_start in range(0, m, batch_size):
                batch_end = min(batch_start + batch_size, m)
                X_batches = X_shuffled[:, batch_start:batch_end] if len(X_shuffled.shape) <= 2 else X_shuffled[batch_start:batch_end, :]
                y_batches = y_shuffled[:, batch_start:batch_end]
                
                # Forward and backward propagation
                y_pred = self.forward_propagation(X_batches)


                batch_loss =  self.configuration['loss'].call(y_pred, y_batches)
                accuracy = self.metric(y_pred, y_batches)
                s = '\rEpoch: [{}] - Loss: [{}]% - Accuracy: [{}]%'.format(epoch, float(np.round(batch_loss, 5)), accuracy*100)

                dout =  self.configuration['loss'].backward(y_pred, y_batches)

                self.backward_propagation(inputs= dout)
                self.update_parameters()

                print(s, end="", flush= True)

            # Calculate average loss for the epoch

            history['loss'].append(batch_loss)

            # Early stopping logic
            if batch_loss < previous_loss:
                patience_counter = 0
                # Save model weights (if applicable)
            else:
                patience_counter += 1

            previous_loss = batch_loss


            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}.")
                break

        return history







    def save_model(self, filepath):
        np.savez(filepath, 
                layers=self.best_parameters,
                layers_type= self.layers_type,
                activations_type= self.activations_type)


    def load_model(self, filepath):
        data = np.load(filepath, allow_pickle=True)
        
        self.layers = list(data['layers'])
        self.activation_names = list(data['activation_names'])





    def metric(self, y_pred, y):

        y_pred = np.argmax(y_pred, axis =0)
        y = np.argmax(y, axis =0)

        return np.mean(y_pred == y)