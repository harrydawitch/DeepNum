import copy
import numpy as np
from numpy.lib.stride_tricks import as_strided


def pick_loss(name, y_pred, y_true, derivative= False):

    
    losses_class = (categorical_crossentropy, 
                    binary_crossentropy, 
                    mean_square_error, 
                    mean_absolute_error)



    if isinstance(name, losses_class):
        if not derivative:
            return name.call(y_pred= y_pred, y_true= y_true)
        else:
            return name.backward(y_pred= y_pred, y_true= y_true)
    


def activate(name):
    activations = {'relu': ReLU,
                   'sigmoid': Sigmoid,
                   'tanh': Tanh,
                   'softmax': Softmax}
    

    if name in activations.keys():
        return activations[name]()
    else:
        raise NameError(f'The activation name provided is incorrect. Please check and re-enter!')
    
    

    
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
    
    


def find_shape(X, mode):       
    threshold = 2
     
    if len(X.shape) > threshold:
        shape = 0 if mode == 'samples' else 1
    else:
        shape = 1 if mode == 'samples' else 0

    return shape




def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """
    Efficient im2col implementation for NHWC format using numpy's as_strided.
    Args:
        input_data: Input data, shape (N, H, W, C)
        filter_h: Height of the filter
        filter_w: Width of the filter
        stride: Stride of convolution
        pad: Padding size
    Returns:
        col: 2D array, shape (N * out_h * out_w, filter_h * filter_w * C)
    """
    N, H, W, C = input_data.shape

    # Add padding
    padded = np.pad(input_data, [(0, 0), (pad, pad), (pad, pad), (0, 0)], mode='constant')
    H_padded, W_padded = padded.shape[1], padded.shape[2]

    # Output dimensions
    out_h = (H_padded - filter_h) // stride + 1
    out_w = (W_padded - filter_w) // stride + 1

    # Stride shapes
    shape = (N, out_h, out_w, filter_h, filter_w, C)
    strides = (
        padded.strides[0],           # batch stride
        padded.strides[1] * stride,  # row stride
        padded.strides[2] * stride,  # column stride
        padded.strides[1],           # filter row stride
        padded.strides[2],           # filter column stride
        padded.strides[3],           # channel stride
    )

    # Create strided view
    patches = as_strided(padded, shape=shape, strides=strides)

    # Reshape to im2col format
    col = patches.reshape(N * out_h * out_w, -1)
    return col



def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    N, H, W, C = input_shape

    # Add padding
    H_padded = H + 2 * pad
    W_padded = W + 2 * pad
    padded = np.zeros((N, H_padded, W_padded, C), dtype=col.dtype)

    # Output dimensions
    out_h = (H_padded - filter_h) // stride + 1
    out_w = (W_padded - filter_w) // stride + 1

    # Reshape col back to patches
    col_reshaped = col.reshape(N, out_h, out_w, filter_h, filter_w, C)

    # Strided view for accumulation
    shape = (N, out_h, out_w, filter_h, filter_w, C)
    strides = (
        padded.strides[0],
        padded.strides[1] * stride,
        padded.strides[2] * stride,
        padded.strides[1],
        padded.strides[2],
        padded.strides[3],
    )
    padded_strided = as_strided(padded, shape=shape, strides=strides)

    # Accumulate values
    np.add.at(padded_strided, (slice(None), slice(None), slice(None), slice(None), slice(None), slice(None)), col_reshaped)

    # Remove padding
    if pad > 0:
        reconstructed = padded[:, pad:-pad, pad:-pad, :]
    else:
        reconstructed = padded

    return reconstructed




def pick_initializer(initializers= 'glorot_uniform', shape= None, name= None):
    if name == 'Dense':
        n_in, n_out = shape
    elif name == 'Conv2D':
        C_out, C_in, H, W = shape
        n_in = C_in * H * W
        n_out = C_out * H * W
    

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
    

from Components.Misc.Losses import categorical_crossentropy, binary_crossentropy, mean_square_error, mean_absolute_error
from Components.Layers.Activations import ReLU, Sigmoid, Tanh, Softmax
from Components.Misc.Metrics import accuracy, precision, recall, f1_score
