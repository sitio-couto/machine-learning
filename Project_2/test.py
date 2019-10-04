
#### DISCLAIMER ####
# Code provided by Orhan Gazi Yalçın (used solely for comparison)
# https://towardsdatascience.com/image-classification-in-10-minutes-with-mnist-dataset-54c35b77a38d


import numpy as np
import tensorflow as tf
import normalization as norm

# Reading stuff
train = np.load('Dataset/train.npz')
valid = np.load('Dataset/val.npz')
x_train, y_train  = train['xs'].astype('float32') , train['ys'].astype('int8')
x_test, y_test = valid['xs'].astype('float32') , valid['ys'].astype('int8')
# x_train = norm.monochrome(x_train, 1024, ch_axis=1)
# x_test = norm.monochrome(x_test, 1024, ch_axis=1)

# Reshaping the array to 4-dims so that it can work with the Keras API
x_train = x_train.reshape(x_train.shape[0], 32, 32, 3)
x_test = x_test.reshape(x_test.shape[0], 32, 32, 3)
input_shape = (32, 32, 3)
# Making sure that the values are float so that we can get decimal points after division
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# Normalizing the RGB codes by dividing it to the max RGB value.
x_train /= 255
x_test /= 255

# Importing the required Keras modules containing model and layers
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
# Creating a Sequential Model and adding the layers
model = Sequential()
model.add(Conv2D(32, kernel_size=(3,3), input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
model.add(Dense(128, activation=tf.nn.relu))
model.add(Dropout(0.2))
model.add(Dense(10,activation=tf.nn.softmax))

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])
model.fit(x=x_train,y=y_train, epochs=10)
print(model.evaluate(x_test, y_test))

#### COLORED RESULTS ####
# E 1 - 66s - L: 1.6789 - Acc: 0.3905
# E 2 - 62s - L: 1.4593 - Acc: 0.4764
# E 3 - 60s - L: 1.3519 - Acc: 0.5146
# E 4 - 58s - L: 1.2635 - Acc: 0.5486
# E 5 - 64s - L: 1.1819 - Acc: 0.5770
# E 6 - 63s - L: 1.1133 - Acc: 0.6023
# E 7 - 64s - L: 1.0415 - Acc: 0.6284
# E 8 - 64s - L: 0.9838 - Acc: 0.6476
# E 9 - 65s - L: 0.9212 - Acc: 0.6718
# E 10 - 65s - L: 0.8610 - Acc: 0.6937
# Val => L: 1.5908 - Acc: 0.5149

#### GRAYSCALE RESULTS ####
# E 1/10 - 55s - L: 1.9341 - Acc: 0.2959
# E 2/10 - 62s - L: 1.7526 - Acc: 0.3659
# E 3/10 - 60s - L: 1.6729 - Acc: 0.3955
# E 4/10 - 59s - L: 1.6111 - Acc: 0.4210
# E 5/10 - 58s - L: 1.5491 - Acc: 0.4431
# E 6/10 - 61s - L: 1.4894 - Acc: 0.4678
# E 7/10 - 60s - L: 1.4287 - Acc: 0.4897
# E 8/10 - 65s - L: 1.3685 - Acc: 0.5108
# E 9/10 - 63s - L: 1.3043 - Acc: 0.5360
# E 10/10 - 61s - L: 1.2493 - Acc: 0.5552
# Val => L: 1.7758 - Acc: 0.3969