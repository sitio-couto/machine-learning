# MC886 - Machine Learning - UNICAMP
# Project 3 - Unsupervised Learning and Dimensionality Reduction
#
# Authors:
# Victor Ferreira Ferrari - RA 187890
# Vinicius Couto Espindola- RA 188115

import numpy as np
from pandas import read_csv
from keras.utils import to_categorical
from const import *
import normalization as norm
import neural as nr
import reduce as red

data = read_csv('Dataset/fashion-mnist_train.csv')
Y = data['label'].to_numpy()
X = data.drop('label', 1).to_numpy()

# Normalization
choice = 2
stats = norm.get_stats(X, choice)
X = norm.normalize_data(X, stats, choice)
Y = to_categorical(Y)

# TODO: run script
# Get first neural network
arc = [300]
dense = nr.get_neural_network_model(arc, (IMG_HEIGHT*IMG_WIDTH, ), N_CLASSES)

# Train network
dense.summary()
dense, hist = nr.train(dense, X, Y, epochs=30, batch_size=128, val_split=0.1)
print('Validation Accuracy: ', np.max(hist['val_accuracy']))

# PCA
variance = [.90, .95, .85]
pca, X = red.reduce_PCA(X, variance[0])
n_comp = pca.n_components_
print('Components:', n_comp)

# Train with PCA
dense = nr.get_neural_network_model(arc, (n_comp, ), N_CLASSES)
dense.summary()
dense, hist = nr.train(dense, X, Y, epochs=30, batch_size=128, val_split=0.1)
print('Validation Accuracy: ', np.max(hist['val_accuracy']))

# Auto-Encoder

# Training

# Auto-Decoder

# Clustering
