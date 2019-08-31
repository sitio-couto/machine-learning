from importlib import import_module
import numpy as np
model = import_module("first_model") 
norm = import_module("normalization")
desc = import_module("gradient_descent")

# Run
#T = desc.normal_equation(X,Y)
# T = desc.descent(X, T, Y)
# print(desc.predict(X,T))
# print(Y)

### Getting Training Set ###
# Returns the data without the header
X, Y, feat_list = model.prepare_dataset("Datasets/training.csv")
# Normalize the features pointed by the received model
X = norm.normalize_data(X, choice=1, features=feat_list)
# Cast, add bias and randomize initial Thetas
X, T, Y = desc.numpy_and_bias(X, Y)

# Validation set
# features to be normalized are the same
X_val, Y_val, _ = model.prepare_dataset("Datasets/validate.csv")
X_val = norm.normalize_data(X_val, choice=1, features=feat_list)
# DO NOT OVERWRITE THETAS
X_val, _, Y_val = desc.numpy_and_bias(X_val, Y_val)

# Train coeficients
Y = np.ravel(Y)
clf = desc.sk_regressor(X, Y)
# Training results
print("Training Score =>", clf.score(X,Y))

# Validate
Y_val = np.ravel(Y_val)
print("Validation Score=>", clf.score(X_val,Y_val))


print(clf)