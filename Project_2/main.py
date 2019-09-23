# MC886 - Machine Learning
# Project 2 - Logistic Regression and Neural Networks
#
# Authors:
# Victor Ferreira Ferrari - RA 187890
# Vinicius Couto Espindola- RA 188115

import numpy as np
import normalization as norm

# Getting Sets
train = np.load('Dataset/train.npz')
X, Y  = train['xs'],train['ys'] 
X = norm.normalize_data(X, choice=1)
print(X)

