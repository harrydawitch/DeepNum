import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core.NeuralNetwork import NN
from Core.Layers import Dense, BatchNormalization, Dropout
from Core.Optimizers import Adam, RMSprop, Momentum,Gradient_Descent
from Core.Activations import ReLU, Softmax
from Core.Regularizers import L1, L2
from Core.Losses import categorical_crossentropy


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
X_test = X_test.T # Now X_test will have shape [784, 10000]

# Convert labels to one-hot encoding
# Each label will be converted to a one-hot encoded vector of size 10
y_train = to_categorical(y_train, 10).T  # Shape will be [10, 60000]
y_test = to_categorical(y_test, 10).T   # Shape will be [10, 10000]



model = NN(X_train, y_train)

model.add(Dense(128, initializer= 'he_uniform', regularizer= L1(alpha= 0.02)))
model.add(BatchNormalization())
model.add(Dropout(0.95))
model.add(ReLU(negative_slope=0.25))
model.add(Dense(64, initializer= 'he_uniform', regularizer= L2(alpha= 0.02)))
model.add(BatchNormalization())
model.add(Dropout(0.95))
model.add(ReLU(negative_slope=0.25))
model.add(Dense(32,  initializer= 'he_uniform'))
model.add(BatchNormalization())
model.add(Dropout(0.95))
model.add(ReLU(negative_slope=0.25))
model.add(Dense(10, initializer= 'he_uniform'))

model.compile(optimizer= 'adam', losses= categorical_crossentropy(logits= True), metrics= 'accuracy')

history = model.train(batch_size= 128, epochs= 10, verbose= True)


