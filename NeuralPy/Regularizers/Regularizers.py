import numpy as np

def l1(alpha= 1, weight= None):
    return alpha * np.sign(weight)

def l2(alpha= 1, weight= None):
    return 2 * alpha * weight

def pick_regularizers(regularizer_args= None, **kwargs):
    if not regularizer_args:
        return 0
    
    # Retrieve regularizer name and lambda value from regularizers class attribute
    regularizer, alpha = regularizer_args
    weight = kwargs['weight']
    
    if isinstance(regularizer,  str) and isinstance (alpha, (int, float)):

        if regularizer == 'l1':
            return l1(alpha= alpha, weight= weight)
        elif regularizer == 'l2':
            return l2(alpha= alpha, weight= weight)
    
    else:
        raise ValueError('Wrong value for regularizer name or alpha!')

