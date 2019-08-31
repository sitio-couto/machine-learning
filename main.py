from importlib import import_module
import numpy as np
norm = import_module("normalization")
desc = import_module("gradient_descent")

# Header was already removed
X, Y = norm.prepare_dataset("Datasets/training.csv")
# Cast, add bias and randomize initial Thetas
X, T, Y = desc.numpy_and_bias(X, Y)
# Run
#T = desc.normal_equation(X,Y)
T = desc.descent(X, T, Y)
print(desc.predict(X,T))
print(Y)

Y = np.ravel(Y)
clf = desc.sk_regressor(X, Y)
print(clf.predict(X))
print(Y)
print(clf.score(X,Y))
#X_val, Y_val = norm.prepare_dataset("Datasets/validate.csv")
#X_val, T, Y_val = desc.numpy_and_bias(X_val, Y_val)
#Y_val = np.ravel(Y_val)
#print(X_val.shape)
#print(Y_val.shape)
#print(clf.score(X_val,Y_val))
