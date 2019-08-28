from importlib import import_module
norm = import_module("normalization")
desc = import_module("gradient_descent")

# THE DATASET STILL HAS ITS HEADER (REMOVE BEFORE REGRESSION)
X, Y = norm.prepare_dataset("Datasets/training.csv")
# Shuffle samples for a proper stochastic gradient
X, Y = desc.shuffle_samples(X, Y)
# Cast and add bias
X, T, Y = desc.numpy_and_bias(X, Y)

# Run
T = desc.descent(X, T, Y)

print(desc.cost(X,T,Y))