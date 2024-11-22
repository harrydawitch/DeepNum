import numpy as np

class categorical_crossentropy:
    def __init__(self):
        self.name = 'categorical_crossentropy'
    
    def call(self, y_pred, y_true, derivative= False):
        m = y_true.shape[1]
        epsilon = 1e-8

        if derivative:
            loss_error = y_pred - y_true
            return loss_error
        
        loss = (- np.sum(y_true * np.log(y_pred + epsilon)) / m)

        return loss 

class binary_crossentropy:
    def __init__(self):
        self.name = 'binary_crossentropy'
    
    def call(self, y_pred, y_true, derivative= False):
        m = y_true.shape[1]  # Number of samples
        epsilon = 1e-8  # Small constant to avoid log(0)

        if derivative:
            loss_error = (y_pred - y_true) * y_pred * (1 - y_pred)
            return loss_error

        # Compute binary cross entropy
        loss = - np.sum(y_true * np.log(y_pred + epsilon) + (1-y_true) * np.log(1-y_pred + epsilon)) / m
        return loss
    
class mean_square_error:
    def __init__(self):
        self.name = 'mse'
    
    def call(self, y_pred, y_true, derivative= False):
        m = y_true.shape[1]  # Number of samples

        if derivative:
            loss_error = (2 * (y_pred - y_true)) / m
            return loss_error

        # Compute binary cross entropy
        loss = (np.sum((y_true - y_pred)**2) / m)
        return loss

class mean_absolute_error:
    def __init__(self):
        self.name = 'mae'
    
    def call(self, y_pred, y_true, derivative= False):
        m = y_true.shape[1]  # Number of samples

        if derivative:
            loss_error = np.where(y_pred > y_true, 1, -1) / m
            return loss_error

        # Compute binary cross entropy
        loss = (np.sum(np.abs(y_true - y_pred)) / m)
        return loss




def pick_loss(name, y_pred, y_true, derivative= False):
    if name == 'categorical_crossentropy':
        return categorical_crossentropy().call(y_pred= y_pred, y_true= y_true, derivative= derivative)
    elif name == 'binary_crossentropy':
        return binary_crossentropy.call(y_pred= y_pred, y_true= y_true, derivative= derivative)
    elif name == 'mse':
        return mean_square_error.call(y_pred= y_pred, y_true= y_true, derivative= derivative)
    elif name == 'mae':
        return mean_absolute_error.call(y_pred= y_pred, y_true= y_true, derivative= derivative)