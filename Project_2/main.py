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
#import neural as nr
import misc

# Getting Sets
train = np.load('Dataset/train.npz')
valid = np.load('Dataset/val.npz')
X, Y  = train['xs'].astype('float32') , train['ys'].astype('int8')
X_v, Y_v = valid['xs'].astype('float32') , valid['ys'].astype('int8')
print("Dataset read!")

# Visualization
#vis.histogram(Y,10)

# Normalization
choice = 1
stats = norm.get_stats(X, choice)
X = norm.normalize_data(X, stats, choice)
print("Training data Normalized!")
X_v = norm.normalize_data(X_v, stats, choice)
print("Val Data normalized!")

# Neural Network (do not apply bias)
#Xn = X.T
#Yn = norm.out_layers(Y)
#vis.learning_curves(Xn, Yn)
#exit(1)

# Initial coefficients and bias.
X = np.insert(X, 0, 1, axis=1)
X_v = np.insert(X_v, 0, 1, axis=1)
print("Bias Added")
classes = np.max(Y) + 1
T = misc.init_coefs(X.shape[1], classes, 57).astype('float32')

# Logistic Regression (Softmax)
print("Regression:")
T = lr.gradient_descent(X, Y, X_v, Y_v, T, 0.001, 10)
v_pred = lr.predict(X_v, T)
print(v_pred, Y_v)

