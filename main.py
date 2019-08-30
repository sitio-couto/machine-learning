from importlib import import_module
import numpy as np
norm = import_module("normalization")
desc = import_module("gradient_descent")

# THE DATASET STILL HAS ITS HEADER (REMOVE BEFORE REGRESSION)
X, Y = norm.prepare_dataset("Datasets/training.csv")
# Shuffle samples for a proper stochastic gradient
X, Y = desc.shuffle_samples(X, Y)
# Cast and add bias
X, T, Y = desc.numpy_and_bias(X, Y)

# Run
T = desc.normal_equation(X,Y)
#T = desc.descent(X, T, Y)
print(desc.cost(X,T,Y))

#Y = np.ravel(Y)
#clf = desc.sk_regressor(X, Y)
#print(X.shape)
#print(Y.shape)
#print(clf.score(X,Y))

#X_val, Y_val = norm.prepare_dataset("Datasets/validate.csv")
#X_val, T, Y_val = desc.numpy_and_bias(X_val, Y_val)
#Y_val = np.ravel(Y_val)
#print(X_val.shape)
#print(Y_val.shape)
#print(clf.score(X_val,Y_val))
