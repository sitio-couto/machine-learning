import numpy as np
from random import randint, sample
from numpy import transpose as tp
from sklearn import linear_model
import time

STEP_LIMIT = 10**(-3) # Step size accepted as convergence
TIME_LIMIT = 10**2       # Descent limit given in seconds
ITER_LIMIT = 10**6    # Maximum amount of iterations for the gradient
MINI_SIZE = 10         # Defines the size for the mini batch
ALPHA = 0.01          # Learning rate

def numpy_and_bias(X, Y):
	'''
		Transforms lists for X and Y into numpy arrays.
		Adds bias to feature matrix (X)
	'''
	
	X = np.array(X)
	Y = np.array(Y)
	X = np.insert(X, 0, 1, axis=1)
	
	return (X,Y)

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
    
    X,Y = numpy_and_bias(X, Y)
    T = np.array(T)
        
    # Setting time limit variables
    end_time = 0
    start_time = time.time()

    # Starting descent
    count = 0
    old_cost = cost(X, T, Y)
    while (end_time - start_time) <= TIME_LIMIT:
        # Getting new Thetas
        T = T - ALPHA*batch_gradient(X, T, Y)
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

def minib_gradient(X, T, Y):
    ''' Returns the gradient calculate using a portion of the samples'''
    m = Y.shape[0]
    b = sample(range(0,(m-1)),MINI_SIZE)
    gradient_vals = (1/MINI_SIZE)*tp(X[b].dot(T) - Y[b]).dot(X[b])
    return tp(gradient_vals)

def stoch_gradient(X, T, Y):
    ''' Returns the gradient calculated using a single random sample.'''
    m = Y.shape[0]
    i = [randint(0,(m-1))] # Select random sample
    gradient_vals = (X[i].dot(T) - Y[i])*tp(X[i])
    return gradient_vals

def sk_regressor(X, Y):
	'''
		Returns the sklearn model fitted.
	    
	    X is the features matrix
	    Y is the target array
	    Both need to be numpy arrays or lists	   
	'''
	
	X,Y = numpy_and_bias(X, Y)

	clf = linear_model.SGDRegressor(max_iter = ITER_LIMIT, tol = STEP_LIMIT, alpha = ALPHA)
	clf.fit(X,Y)
	return clf

def normal_equation(X, Y):
	'''
		Returns the analytical solution via normal equation.
		
		X is the features matrix
		Y is the target array
		Both need to be numpy arrays or lists	
	'''
	# If the data is not a numpy array. 
	X,Y = numpy_and_bias(X,Y)
	
	# Normal equation: step 1
	square = X.T.dot(X)
	
	# Check if matrix is invertible
	if np.linalg.det(square) == 0:
		print("Matrix not invertible! Cannot be solved by normal equation.")
		return None
	
	# Rest of equation
	theta = ((np.linalg.inv(square)).dot(X.T)).dot(Y)
	return theta


# Two ways of writing the gradient calculation
# (1/m)*tp(X).dot(X.dot(T) - Y)
# tp((1/m)*tp(X.dot(T) - Y).dot(X))

# # Sample 1 => Unsolvable
# X = np.array([[1, 2, 3],[4, 5, 6],[7, 8, 9]])
# Y = np.array([[3],[6],[9]])
# T = np.array([[2],[4],[6]])

# # Sample 2 => Expects T = ( 1,-1, 3)
# X = np.array([[4, -1, 1],[2, 5, 2],[1, 2, 4]])
# Y = np.array([[8],[3],[11]])
# T = np.array([[2],[4],[6]])

# Sample 3 => Large generated sample
f = 40 # amount of features
m = 50 # amount of samples
X = [[i*0.01 for i in sample(range(-100,100),f)] for x in range(m)]
Y = [[i for i in sample(range(-100,100),1)] for x in range(m)]
T = [[i for i in sample(range(-20,20),1)] for x in range(f)]
X = np.array(X)
Y = np.array(Y)
T = np.array(T)

T = descent(X, T, Y)
print(T)
print("Cost =>",cost(X,T,Y))
