import numpy as np

def initializer(initializers= None, shape= None):
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

