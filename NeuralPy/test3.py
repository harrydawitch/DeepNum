import numpy as np


def normalize(X):
    epsilon = 1e-8
    m = X.shape[1]
    mean = np.sum(X, axis=1, keepdims= True) / m
    var =  np.sum((X - mean)**2, axis= 1, keepdims= True) / m
    x_normalized = (X - mean) / np.sqrt(var + epsilon)
    return x_normalized

X = np.array([[1, 2, 3],
              [4, 5, 6]])  # Shape (2, 3)

normalized_X = normalize(X)
print(normalized_X)
