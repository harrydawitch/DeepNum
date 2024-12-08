import numpy as np
from .Utils import activate, find_shape

class categorical_crossentropy:
    def __init__(self, logits= False):
        self.name = 'categorical_crossentropy'
        self.logits= logits

    def call(self, y_pred, y_true):
        m = y_true.shape[find_shape(y_true, mode='samples')]
        epsilon = 1e-8
        
        if self.logits:
            activation = activate('softmax')
            y_pred = activation.forward(y_pred)

        loss = (- np.sum(y_true * np.log(y_pred + epsilon)) / m)
        return loss 
    
    def backward(self, y_pred, y_true):
        epsilon = 1e-8
        if self.logits:
            activation = activate('softmax')
            predicted_proba = activation.forward(y_pred)
            loss_error =  predicted_proba - y_true      
            return loss_error
        else:
            loss_error = y_pred - y_true
            return loss_error


#===========================================================================================================================================================


class binary_crossentropy:
    def __init__(self, logits = False):
        self.name = 'binary_crossentropy'
        self.logits= logits

    def call(self, y_pred, y_true):
        m = y_true.shape[find_shape(y_true, mode='samples')]  # Number of samples
        epsilon = 1e-8  # Small constant to avoid log(0)

        # Compute binary cross entropy
        loss = - np.sum(y_true * np.log(y_pred + epsilon) + (1-y_true) * np.log(1-y_pred + epsilon)) / m
        return loss

    def backward(self, y_pred, y_true):
        if self.logits:
            activation = activate('sigmoid')
            predicted_proba = activation.forward(y_pred)
            loss_error = predicted_proba - y_true
            return loss_error

        else:
            loss_error = (y_pred - y_true) / (y_pred * (1 - y_pred))
            return loss_error


#===========================================================================================================================================================


class mean_square_error:
    def __init__(self):
        self.name = 'mse'
    
    def call(self, y_pred, y_true):
        m = y_true.shape[find_shape(y_true, mode='samples')]  # Number of samples

        # Compute binary cross entropy
        loss = np.mean((y_true - y_pred)**2, axis= 1)
        return loss

    def backward(self, y_pred, y_true):
        m = y_true.shape[find_shape(y_true, mode='samples')]  # Number of samples
        (2 * (y_pred - y_true)) / m
        return (2 * (y_pred - y_true)) / m


#===========================================================================================================================================================


class mean_absolute_error:
    def __init__(self):
        self.name = 'mae'
    
    def call(self, y_pred, y_true):
        m = y_true.shape[find_shape(y_true, mode='samples')]  # Number of samples

        # Compute binary cross entropy
        loss = (np.sum(np.abs(y_true - y_pred)) / m)
        return loss

    def backward(self, y_pred, y_true):
        m = y_true.shape[find_shape(y_true, mode='samples')]  # Number of samples
        loss_error = np.sign(y_pred - y_true) / m
        return loss_error


