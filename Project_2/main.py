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
classes = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
print("Dataset read!")

# Normalization
choice = 2
# X = norm.monochrome(X, 1024, ch_axis=1)######################################################
# X_v = norm.monochrome(X_v, 1024, ch_axis=1)######################################################
stats = norm.get_stats(X, choice)
X = norm.normalize_data(X, stats, choice).astype('float32')
print("Training Data Normalized!")
X_v = norm.normalize_data(X_v, stats, choice).astype('float32')
print("Validation Data Normalized!")

#### MULTINOMIAL LOGISTIC REGRESSION ####
#run.logistic(X, X_v, Y, Y_v, 0.01, 300, classes)

#### NEURAL NETWORK ####
run.neural_network(X, X_v, Y, Y_v, classes)

#### VISUALIZATION ####
#run.visualize_data(X, Y, X_v, Y_v)
