from Core.NeuralPy import *
from Losses.Losses import *
from Layers.Dense import *
from Layers.Dropout import *
from Optimizers.Optimizers import *
from Activations.Activations import *
from Layers.BatchNormalization import *

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




model = NeuralPy()

model.add(Dense(128, activation= 'relu', input_size= X_train.shape[0], initializer= 'he_uniform', regularizer= ('l2', 0.001)))
model.add(BatchNormalization())
model.add(Dense(64, activation= 'relu', initializer= 'he_uniform', regularizer= ('l2', 0.001)))
model.add(BatchNormalization())
model.add(Dense(32, activation= 'relu', initializer= 'he_uniform', regularizer= ('l2', 0.001)))
model.add(BatchNormalization())
model.add(Dense(10, activation= 'softmax', initializer= 'he_uniform'))

model.compile(optimizer= Adam(learning_rate= 0.001, clip= 4.0), losses= 'categorical_crossentropy', metrics= 'accuracy')

history = model.fit(X_train, y_train, batch_size= 128, epochs= 10, verbose= True)

prediction = model.predict(X_test)
print(prediction)

