import numpy as np
from random import randint, sample
from sklearn import linear_model
import time

STEP_LIMIT = 10**(-3) # Step size accepted as convergence
TIME_LIMIT = 180    # Descent limit given in seconds
ITER_LIMIT = 10**6    # Maximum amount of iterations for the gradient
MINI_SIZE = 2         # Defines the size for the mini batch
ALPHA = 0.1          # Learning rate

def numpy_and_bias(X, Y, T_set=(-10,10)):
	'''
		Transforms lists for X and Y into numpy arrays.
		Adds bias to feature matrix (X)
	'''
	
	X = np.array(X)
	Y = np.array(Y)
	X = np.insert(X, 0, 1, axis=1)
	T = np.array([[randint(-T_set[0],T_set[1])] for x in range(X.shape[1])])

	return (X,T,Y)

def cost(X, T, Y):
    '''Returns the cost function value for the given set of variables.'''
    m = Y.shape[0]
    cost = (1/(2*m))*sum((X.dot(T) - Y)**2)
    return cost[0] 

def shuffle_samples(X, Y):
    '''Return samples in randomized order'''
    X = np.concatenate((X,Y),axis=1)
    np.random.shuffle(X)
    Y = X[:,[-1]]
    X = X[:,:-1]
    return X, Y 

def descent(X, T, Y, type='s'):
    '''
        Return the models convergence obtained by the gradient descent.
        It is assumed that the bias is already included and the samples are shuffled.

        Parameters:
            X (Float 2dArray): The coeficient matrix.
            T (Float 2dArray): The variables matrix (to be modified).
            Y (Float 2dArray): The results matrix.
            type (int): The choice of descent ('s'-stoch|'m'-mini|'b'-batch).

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
        if (type=='b'):
            delta = batch_gradient(X, T, Y)
        elif (type=='m'):
            delta = minib_gradient(X, T, Y)
        else:
            delta = batch_gradient(X, T, Y)
        
        T = T - ALPHA*delta
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
    gradient_vals = (1/m)*(X.dot(T) - Y).T.dot(X)
    return (gradient_vals).T

def minib_gradient(X, T, Y):
    ''' Returns the gradient calculate using a portion of the samples'''
    # Get index of first sample for the mini batch
    base = minib_gradient.count
    m = Y.shape[0]
    b = list(range(base, min((base + MINI_SIZE), m)))
    gradient_vals = (1/MINI_SIZE)*(X[b].dot(T) - Y[b]).T.dot(X[b])

    # Update samples for mini batch
    minib_gradient.count += MINI_SIZE
    if minib_gradient.count >= m:
        minib_gradient.count = 0 

    return (gradient_vals).T

def stoch_gradient(X, T, Y):
    ''' Returns the gradient calculated using a single random sample.'''
    m = Y.shape[0]
    i = [stoch_gradient.count] # Get next sample
    gradient_vals = (X[i].dot(T) - Y[i])*(X[i]).T

    # Update sample for stochastic
    stoch_gradient.count += 1
    if stoch_gradient.count >= m:
        stoch_gradient.count = 0 

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

### Static variables ####
# References the current sample(s) used, ensuring that gradients
# such as stoch and minib iterate through all the samples instead
# of repeating samples in each iteration.
minib_gradient.count = 0
stoch_gradient.count = 0
#########################

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

# # Sample 3 => Large generated sample
# f = 40 # amount of features
# m = 50 # amount of samples
# X = [[i*0.01 for i in sample(range(-100,100),f)] for x in range(m)]
# Y = [[i for i in sample(range(-100,100),1)] for x in range(m)]
# T = [[i for i in sample(range(-20,20),1)] for x in range(f)]
# X = np.array(X)
# Y = np.array(Y)
# T = np.array(T)

# Add bias to features and generate the first set of T vals
# X,T,Y = numpy_and_bias(X,Y)

# Shuffle samples to randomize minib and stoch gradients
# X, Y = shuffle_samples(X,Y)



# T = descent(X, T, Y)
# print(T)
# print("Cost =>",cost(X,T,Y))
