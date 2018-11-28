# Implementation of LeNet-5 in keras
# [LeCun et al., 1998. Gradient based learning applied to document recognition]
# Some minor changes are made to the architecture like using ReLU activation instead of
# sigmoid/tanh, max pooling instead of avg pooling and softmax output layer

import keras
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import MaxPooling2D
from keras.layers import Conv2D
from keras.models import Sequential
from keras.utils.np_utils import to_categorical
from keras import backend as K
import tensorflowjs as tfjs
import numpy as np
import pandas as pd
import cv2

dataset = pd.read_csv('./full_single.csv')
x_train = dataset.iloc[:, 1:-1]
y_train = train.iloc[0]

X_train = np.array(X_train)
X_test = np.array(X_test)

# Reshape the training and test set
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

# Standardization
mean_px = X_train.mean().astype(np.float32)
std_px = X_train.std().astype(np.float32)
X_train = (X_train - mean_px)/(std_px)

# One-hot encoding the labels
Y_train = to_categorical(Y_train)


model = Sequential()
# Layer 1
# Conv Layer 1
model.add(Conv2D(64,
                 kernel_size=7,
                 strides=3,
                 activation='relu',
                 input_shape=(32, 32, 1)))
# Pooling layer 1
model.add(MaxPooling2D(pool_size=3, strides=2))
# Layer 2
# Conv Layer 2
model.add(Conv2D(128,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 input_shape=(14, 14, 6)))
# Pooling Layer 2
model.add(MaxPooling2D(pool_size=3, strides=2, ou))
# Flatten
model.add(Flatten())
# Layer 3
# Fully connected layer 1
model.add(Dense(units=120, activation='relu'))
# # Layer 4
# # Fully connected layer 2
# model.add(Dense(units=84, activation='relu'))
# Layer 5
# Output Layer
model.add(Dense(units=10, activation='softmax'))
model.compile(optimizer='', loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, Y_train, steps_per_epoch=10, epochs=42)

y_pred = model.predict(X_test)

# Converting one hot vectors to labels
labels = np.argmax(y_pred, axis=1)

index = np.arange(1, 28001)

labels = labels.reshape([len(labels), 1])
index = index.reshape([len(index), 1])

final = np.concatenate([index, labels], axis=1)

# Prediction csv file
np.savetxt("mnist_1.csv", final, delimiter=" ", fmt='%s')
