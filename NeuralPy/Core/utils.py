from Core.Metrics import *
from Core.Regularizers import *
import copy

def pick_loss(name, y_pred, y_true, derivative= False):
    from Core.Losses import categorical_crossentropy, binary_crossentropy, mean_square_error, mean_absolute_error
    
    loss = {'categorical_crossentropy': categorical_crossentropy,
            'binary_crossentropy': binary_crossentropy,
            'mse': mean_square_error,
            'mae': mean_absolute_error}
    
    losses_class = (categorical_crossentropy, binary_crossentropy, mean_square_error, mean_absolute_error)

    
    # if name in loss.keys() or name in loss.values():

    #     if not derivative:
    #         return loss[name]().call(y_pred= y_pred, y_true= y_true)
    #     else:
    #         return loss[name]().backward(y_pred= y_pred, y_true= y_true)
    
    # else: 
    #     raise NameError('Wrong name for loss function')
    

    if isinstance(name, losses_class):
        if not derivative:
            return name.call(y_pred= y_pred, y_true= y_true)
        else:
            return name.backward(y_pred= y_pred, y_true= y_true)

    elif isinstance(name, str):
        if not derivative:
            return loss[name].call(y_pred= y_pred, y_true= y_true)
        else:
            return loss[name].backward(y_pred= y_pred, y_true= y_true)
    else:
        raise ValueError('Wrong name for loss function. Please check again!')
    

def activate(name):
    from Core.Activations import ReLU, Sigmoid, Tanh, Softmax

    activations = {'relu': ReLU,
                'sigmoid': Sigmoid,
                'tanh': Tanh,
                'softmax': Softmax}
    

    if name in activations.keys():
        return activations[name]()
    else:
        raise NameError(f'The activation name provided is incorrect. Please check and re-enter!')
    
    

def pick_initializer(initializers= None, shape= None):
    n_in, n_out = shape
    # Compute and return the regularization term if it meet the condition

    if initializers == 'he_normal':
        return np.random.normal(loc= 0.0, scale= np.sqrt(2./n_in), size= shape)

    elif initializers == 'he_uniform':
        limit = np.sqrt(6./n_in)
        return np.random.uniform(low= -limit, high= limit, size= shape)

    elif initializers == 'glorot_normal':
        return np.random.normal(loc= 0.0, scale= np.sqrt(2./(n_in + n_out)), size= shape)

    elif initializers == 'glorot_uniform':
        limit = np.sqrt(6./(n_in + n_out))
        return np.random.uniform(low= -limit, high= limit, size= shape)
    

def pick_metric(metrics= None, y_pred= None, y_true= None):
    # Dictionary to map metric names to their respective methods
    metric_dict = {'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1_score}

    # Check if a compiled metric exists
    if metrics is not None:

        # Check if the requested metric is in the dictionary of available metrics then call its function
        if metrics in metric_dict:
            return metric_dict[metrics](y_pred= y_pred, y= y_true)
        
        else:
            # Raise an error if the metric name is not valid
            raise ValueError(f'No such metric name {metrics}')
    else:
        # Raise an error if no metric has been compiled
        raise ValueError(f'Metric has not been compiled')
    

def pick_optimizer(name, layer, **kwargs):
    from Core.Optimizers import Adam, Gradient_Descent, Momentum, RMSprop

    optimizer = {'adam': Adam,
                 'sgd': Gradient_Descent,
                 'momentum': Momentum,
                 'rmsprop': RMSprop}
    
    if name in optimizer.keys():
        layer.optimizer = optimizer[name]()
        return layer.optimizer.call(layer= layer)
    else:
        raise NameError(f'The optimizer name provided is incorrect. Please check and re-enter!')
    
    

def update_parameter(optimizer, layer):
    from Core.Optimizers import Adam, Gradient_Descent, Momentum, RMSprop


    optimizer_classes = (Adam, RMSprop, Momentum, Gradient_Descent)

    if layer.optimizer is None:


        if isinstance(optimizer, optimizer_classes):
            layer.optimizer = copy.deepcopy(optimizer)
            layer.optimizer.call(layer= layer)

        elif isinstance(optimizer, str):
            pick_optimizer(name= optimizer, layer= layer)

        else:
            raise ValueError('You are enter wrong optimizer object type. Please check again!')
        
    else:
        layer.optimizer.call(layer= layer)
    