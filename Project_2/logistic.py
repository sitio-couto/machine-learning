import numpy as np

# Activation function
def sigmoid(x):
    return 1/(1+np.exp(-x))

def hypothesis(X, T):
    return sigmoid(X.dot(T))


# Cost function
def cost(Y, Y_pred):
    return (-1 * Y * log(Y_pred) - (1-Y) * log(1-Y_pred)).mean()

def cost_derivative(X, Y, Y_pred):
    error = Y_pred - Y
    return X.T.dot(error)

def log(x, bound=1e-16):
    return np.log(np.maximum(x,bound))

# Gradient Descent
def gd_step(X, Y, T, Y_pred, alpha):
    return T - alpha * cost_derivative(X, Y, Y_pred)
