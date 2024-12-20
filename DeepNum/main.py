import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical


from Components.Model import Sequential
from Components.Layers.Dense import Dense
from Components.Layers.Conv2D import Conv2D
from Components.Layers.Flatten import Flatten
from Components.Layers.BatchNormalization import BatchNormalization
from Components.Layers.Activations import ReLU, Softmax
from Components.Layers.Pooling import MaxPooling
from Components.Layers.Dropout import Dropout

from Components.Optimizers import Adam, Gradient_Descent
from Components.Losses import categorical_crossentropy




# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = np.expand_dims(x_train, axis= -1).transpose(0, 3, 1, 2)
x_test =  np.expand_dims(x_test, axis= -1).transpose(0, 3, 1, 2)

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)


model = Sequential(X_train= x_train, y_train= y_train)


model.add(Conv2D(n_filters= 16, filter_size= 3, stride= 1, padding= 1, initializer= 'he_normal'))
model.add(ReLU())








model.add(Flatten())

model.add(Dense(units= 100))
model.add(BatchNormalization())
model.add(ReLU())
model.add(Dropout(0.6))
model.add(Dense(units= 10))

model.config(optimizer= Adam(clip= 3.0), loss= categorical_crossentropy(logits= True))

model.train(batch_size= 64, epochs= 10)