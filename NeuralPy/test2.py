import numpy as np
np.random.seed(32)
rows = 3
cols = 3
channels = 1
stride = 0


inputs = np.random.randint(1, 50, size= (rows, cols))
kernel = np.random.randint(1, 9, size = (5, 3, 3))

b = np.pad(inputs, pad_width= 0, mode= 'constant', constant_values= 0)
print(b)