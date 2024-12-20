import copy
import numpy as np

    


def find_shape(X, mode):       
    threshold = 2
     
    if len(X.shape) > threshold:
        shape = 0 if mode == 'samples' else 1
    else:
        shape = 0 if mode == 'samples' else 1

    return shape



def extracted_patches(shape, filter_h, filter_w, stride= 1, pad= 0):
    N, C, H, W = shape


    out_h = ((H - filter_h + (2 * pad)) // stride) + 1
    out_w = ((W - filter_w + (2 * pad)) // stride) + 1

    i0 = np.repeat(np.arange(filter_h ,dtype='int32'), filter_w)

    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(out_h, dtype='int32'), out_w)

    j0 = np.tile(np.arange(filter_w), filter_h * C)
    j1 = stride * np.tile(np.arange(out_w, dtype='int32'), int(out_h))

    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(C, dtype='int32'), filter_h * filter_w).reshape(-1, 1)

    return (k, i, j)



def im2col(X, filter_h, filter_w, stride=1, padding=0):

    x_padded = np.pad(X, ((0,0), (0,0), (padding, padding), (padding, padding)), mode= 'constant')

    k, i, j = extracted_patches(X.shape, filter_h, filter_w, stride, padding)

    cols = x_padded[:, k, i, j]
    C = X.shape[1]

    cols = cols.transpose(1, 2, 0).reshape(filter_h * filter_w * C, -1)
    return cols



def col2im(cols, x_shape, filter_h, filter_w, stride=1, padding=1):

  N, C, H, W = x_shape

  H_padded, W_padded = H + 2 * padding, W + 2 * padding
  x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)

  k, i, j = extracted_patches(x_shape, filter_h, filter_w, stride, padding)

  cols_reshaped = cols.reshape(C * filter_h * filter_w, -1, N)
  cols_reshaped = cols_reshaped.transpose(2, 0, 1)

  np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)

  if padding == 0: return x_padded

  return x_padded[:, :, padding:-padding, padding:-padding]




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
    

def get_output_pad(X, filter_h, filter_w, stride, padding):
    assert stride == 0

    _, _, H, W = X.shape
    
    out_h = ((H - filter_h + (2 * padding)) // stride) + 1
    out_w = ((W - filter_w + (2 * padding)) // stride) + 1

    return out_h, out_w



def get_batches(batch_size, inputs):
    pass

    

