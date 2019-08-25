import numpy as np
from random import randint
from numpy import transpose as tp
import time

STEP_LIMIT = 10**(-3) # Step size accepted as convergence
TIME_LIMIT = 10       # Descent limit given in seconds
ITER_LIMIT = 10**5    # Maximum amount of iterations for the gradient
ALPHA = 0.0001        # Learning rate

def cost(X, T, Y):
    '''Returns the cost function value for the given set of variables.'''
    m = Y.shape[0]
    cost = (1/(2*m))*sum((X.dot(T) - Y)**2)
    return cost[0] 

def descent(X, T, Y):
    '''
        Return the models convergence obtained by the gradient descent.

        Parameters:
            X (Float 2dArray): The coeficient matrix.
            T (Float 2dArray): The variables matrix (to be modified).
            Y (Float 2dArray): The results matrix.

        Returns:
            T (Float 2dArray): The minimized cost coeficients obtained for the model.
    '''
    # Setting time limit variables
    end_time = 0
    start_time = time.time()

    # Starting descent
    count = 0
    old_cost = cost(X, T, Y)
    while (end_time - start_time) <= TIME_LIMIT:
        # Getting new Thetas
        T = T - ALPHA*stoch_gradient(X, T, Y)
        count += 1 # Count iteration

        # Updating hyperparameters
        new_cost = cost(X, T, Y)
        step = abs(old_cost - new_cost)
        old_cost = new_cost
        end_time = time.time()

        # Testing gradient termination
        if count >= ITER_LIMIT:
            return T

    print("NOTE: Time limit for descent exceded.")
    return T

def batch_gradient(X, T, Y):
    ''' Returns the gradiente calculated using all samples.'''
    m = Y.shape[0]
    gradient_vals = (1/m)*tp(X.dot(T) - Y).dot(X)
    return tp(gradient_vals)

# def minib_gradient(X, T, Y):

def stoch_gradient(X, T, Y):
    ''' Returns the gradient calculated using a single random sample.'''
    m = Y.shape[0]
    i = randint(0,(m-1)) # Select random sample
    gradient_vals = (X[[i]].dot(T) - Y[i])*tp(X[[i]])
    return gradient_vals

# Two ways of writing the gradient calculation
# (1/m)*tp(X).dot(X.dot(T) - Y)
# tp((1/m)*tp(X.dot(T) - Y).dot(X))

# # Sample 1 => Unsolvable
# X = np.array([[1, 2, 3],[4, 5, 6],[7, 8, 9]])
# Y = np.array([[3],[6],[9]])
# T = np.array([[2],[4],[6]])

# Sample 2 => Expects T = ( 1,-1, 3)
X = np.array([[4, -1, 1],[2, 5, 2],[1, 2, 4]])
Y = np.array([[8],[3],[11]])
T = np.array([[2],[4],[6]])

T = descent(X, T, Y)
print(T)
print("Cost =>",cost(X,T,Y))