# MC886 - Machine Learning - UNICAMP
# Project 2 - Logistic Regression and Neural Networks
#
# Authors:
# Victor Ferreira Ferrari - RA 187890
# Vinicius Couto Espindola- RA 188115

import numpy as np
import normalization as norm
import visualization as vis
import logistic as lr

# Getting Sets
train = np.load('Dataset/train.npz')
valid = np.load('Dataset/val.npz')
X, Y  = train['xs'],train['ys']
X_v, Y_v = valid['xs'],valid['ys']

# Visualization
#vis.histogram(Y,10)

# Normalization
choice = 1
stats = norm.get_stats(X, choice)
X = norm.normalize_data(X, stats, choice)
X_v = norm.normalize_data(X_v, stats, choice)

# Logistic Regression (Softmax)
