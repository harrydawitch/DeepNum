import numpy as np
import copy




class NN(Container):
    def __init__(self, X_train= None, y_train= None):
        if X_train is None or y_train is None:
            raise ValueError(f'Missing input! neither X_train nor y_train')
        
        super().__init__()
        self.input = {'X_train': X_train,
                      'y_train': y_train}
        



    def add(self, layer):
        self.layers.append(layer)
        super().setup_layers(layer)


    def train(self,  batch_size= None, epochs= None, verbose= False, show_metric= False, patience= float('inf')):
        
        m = self.input['X_train'].shape[find_shape(self.input['X_train'], mode= 'samples')]

        if not batch_size:
            batch_size = m
        
        if not epochs:
            raise KeyError('Epoch is not specify')
        
        
        # Call the parent class method to setup layers by assigning parameters to each layers
        super().setup_layers()


        previous_loss = float('inf')
        permutation = np.random.permutation(m)
        patience_counter = 0
        history = {'loss': [],
                   'accuracy': []}


        for epoch in range(1, epochs + 1):
            X_shuffled = self.input['X_train'][:, permutation]
            y_shuffled = self.input['y_train'][:, permutation]
            total_batches_loss = 0 

            for batch in range(0, m, batch_size):
                X_batches = X_shuffled[:, batch:batch + batch_size] 
                y_batches = y_shuffled[:, batch:batch + batch_size]

                batch_samples = X_batches.shape[find_shape(X_batches, mode= 'samples')]

                y_pred = super().forward_propagation(X_batches)
                super().backward_propagation(y_batches)

                batch_loss = pick_loss(name= self.losses, y_pred= y_pred, y_true= y_batches, derivative= False)
                total_batches_loss += (batch_loss * batch_samples)
            
            epoch_loss = total_batches_loss / m
            history['loss'].append(epoch_loss)

            if verbose:
                print(f'Epoch {epoch}/{epochs} - Loss: {epoch_loss}')

            if show_metric:
                prediction = self.forward_propagation(self.input['X_train'])
                metric_value = pick_metric(metrics= self.metrics, y_pred= prediction, y_true= self.input['y_train'] )
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


