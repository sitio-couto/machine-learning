# MC886 - Machine Learning - UNICAMP
# Project 2 - Logistic Regression and Neural Networks
#
# Authors:
# Victor Ferreira Ferrari - RA 187890
# Vinicius Couto Espindola- RA 188115

import numpy as np
import normalization as norm
import visualization as visu
import run

# Getting Sets
train = np.load('Dataset/train.npz')
valid = np.load('Dataset/val.npz')
X, Y  = train['xs'].astype('float32') , train['ys'].astype('int8')
X_v, Y_v = valid['xs'].astype('float32') , valid['ys'].astype('int8')
classes = np.max(Y) + 1
print("Dataset read!")

# Normalization
choice = 2
stats = norm.get_stats(X, choice)
X = norm.normalize_data(X, stats, choice).astype('float32')
print("Training data Normalized!")
X_v = norm.normalize_data(X_v, stats, choice).astype('float32')
print("Val Data normalized!")

#### MULTINOMIAL LOGISTIC REGRESSION ####
# run.logistic(X, X_v, Y, Y_v, 0.01, 300, classes)

#### NEURAL NETWORK ####
#run.neural_network(X, Y)

#### PLOTTING ####
X = X.T
Y = norm.out_layers(Y.T)
X_v = X_v.T
Y_v = norm.out_layers(Y_v.T)
print(X.shape, Y.shape)
visu.learning_curves(X, Y, X_v, Y_v, n=(0,3073), m=10000)