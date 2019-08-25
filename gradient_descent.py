import numpy as np
from numpy import transpose as tp
import time

STEP_LIMIT = 10**(-6) # Step size accepted as convergence
TIME_LIMIT = 10    # Descent limit given in seconds
ALPHA = 0.00001         # Learning rate

def cost(X, T, Y):
    m = Y.shape[0]
    cost = (1/(2*m))*sum((X.dot(T) - Y)**2)
    return cost[0] 

def descent(X, T, Y):
    # Setting time limit variables
    end_time = 0
    start_time = time.time()

    # Starting descent
    old_cost = cost(X, T, Y)
    while (end_time - start_time) <= TIME_LIMIT:
        # Getting new Thetas
        T = T - ALPHA*gradient(X, T, Y)
        
        # Updating hyperparameters
        new_cost = cost(X, T, Y)
        step = abs(old_cost - new_cost)
        old_cost = new_cost
        end_time = time.time()

        # Testing convergence
        if step <= STEP_LIMIT : 
            return T

    print("NOTE: Time limit for descent exceded.")
    return T

def gradient(X, T, Y):
    m = Y.shape[0]
    gradient_vals = (1/m)*tp(X.dot(T) - Y).dot(X)
    return tp(gradient_vals)

# (1/m)*tp(X).dot(X.dot(T) - Y)
# tp((1/m)*tp(X.dot(T) - Y).dot(X))

# Sample 1 => Unsolvable
X = np.array([[1, 2, 3],[4, 5, 6],[7, 8, 9]])
Y = np.array([[3],[6],[9]])
T = np.array([[2],[4],[6]])

# # Sample 2 => Expects T = ( 1,-1, 3)
# X = np.array([[4, -1, 1],[2, 5, 2],[1, 2, 4]])
# Y = np.array([[8],[3],[11]])
# T = np.array([[2],[4],[6]])

T = descent(X, T, Y)
print(T)
print("Cost =>",cost(X,T,Y))