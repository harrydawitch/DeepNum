from Core.NeuralPy import *
from Losses.Losses import *
from Layers.Dense import *
from Optimizers.Optimizers import *
from Activations.Activations import *
from test3 import DNN

import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# Load the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Reshape and normalize input data
# Flatten the 28x28 images to 784 features and normalize to the range [0, 1]
X_train = X_train.reshape(-1, 784).astype("float32") / 255  # Shape will be [60000, 784]
X_test = X_test.reshape(-1, 784).astype("float32") / 255    # Shape will be [10000, 784]

# Convert the data to the format [features, examples]
X_train = X_train.T  # Now X_train will have shape [784, 60000]
X_test = X_test.T    # Now X_test will have shape [784, 10000]

# Convert labels to one-hot encoding
# Each label will be converted to a one-hot encoded vector of size 10
y_train = to_categorical(y_train, 10).T  # Shape will be [10, 60000]
y_test = to_categorical(y_test, 10).T   # Shape will be [10, 10000]


model = DNN()

model.Dense(128, input_dim= X_train.shape[0], activation= 'relu', initializers= 'he_normal', regularizer=('l2',0.2))
model.Dense(64, activation= 'relu', initializers= 'he_normal', regularizer=('l2',0.2))
model.Dense(32, activation= 'relu', initializers= 'he_normal', regularizer=('l2',0.2))
model.Dense(10, activation= 'softmax', initializers= 'he_uniform')

model.compile(optimizer= 'adam', losses= 'categorical_crossentropy', metrics= 'accuracy')

history = model.fit(X_train, y_train, batch_size= 32, epochs= 15, verbose= True)