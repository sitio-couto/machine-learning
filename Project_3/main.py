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
import clustering as clus
import visualization as vis
import misc
import run

data = read_csv('Dataset/fashion-mnist_train.csv')
Y = data['label'].to_numpy()
X = data.drop('label', 1).to_numpy()
classes = list(CLASS_NAMES.values())

# Normalization
choice = 2
stats = norm.get_stats(X, choice)
X = norm.normalize_data(X, stats, choice).astype('float32')
Y = to_categorical(Y).astype('int8')

# Get first neural network
arc = [300]
run.run_network(X, Y, arc, (IMG_HEIGHT*IMG_WIDTH, ), N_CLASSES , epochs=30, batch_size=1024, val_split=0.1)

# PCA
variance = [.90, .95, .85]
pca, Xpca = red.reduce_PCA(X, variance[0])
n_comp = pca.n_components_
print('Components:', n_comp)

# Train with PCA
run.run_network(Xpca, Y, arc, (n_comp, ), N_CLASSES , epochs=30, batch_size=1024, val_split=0.1)

# Dense Auto-Encoder
arc_enc = [128, 64, 32]
autoenc, enc = red.autoencoder(X, arc_enc, (IMG_HEIGHT*IMG_WIDTH, ))
autoenc, _ = nr.train(autoenc, X, X, epochs=50, batch_size=1024, val_split=0, use_calls=False, best=False)
Xenc = enc.predict(X)

# Training
run.run_network(Xenc, Y, arc, (arc_enc[-1], ), N_CLASSES , epochs=30, batch_size=1024, val_split=0.1)

# Convolutional Auto-Encoder
Ximg = X.reshape((X.shape[0], IMG_HEIGHT, IMG_WIDTH))

# K-Means
km, pred = clus.k_means(Xpca, N_CLASSES, 'k-means++', 1000, 1e-4)
vis.histogram(pred, N_CLASSES, 'K-Means')

# OPTICS
opt, pred = clus.optics(Xpca, 2)
print(pred)
vis.histogram(pred, N_CLASSES, 'OPTICS')
