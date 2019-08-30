import numpy as np
from random import randint, sample, shuffle
from sklearn import linear_model
import time

# STEP_LIMIT = 10**(-3) # Step size accepted as convergence
TIME_LIMIT = 5   # Descent limit given in seconds
EPOCHS_LIMIT = 10**4    # Maximum amount of iterations for the gradient
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

def descent(X, T, Y, type='b'):
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

    # Starting descent
    boot_epoch_data(X, T, Y)
    while (time.time() - start_time) <= TIME_LIMIT:

        # Getting new Thetas
        if (type=='b'):
            delta = batch_gradient(X, T, Y)
        elif (type=='m'):
            delta = minib_gradient(X, T, Y)
        else:
            delta = stoch_gradient(X, T, Y)
        
        # Update gradients
        T = T - ALPHA*delta 

        # Check termination
        if epochs_count >= EPOCHS_LIMIT: 
            return T
        
    print("NOTE: Time limit for descent exceded.")
    return T

def batch_gradient(X, T, Y):
    ''' Returns the gradiente calculated using all samples.'''
    m = Y.shape[0]
    gradient_vals = (1/m)*(X.dot(T) - Y).T.dot(X)
   
    # Update epoch global vars
    update_epoch(X, T, Y, m, m)

    return (gradient_vals).T

def minib_gradient(X, T, Y):
    ''' Returns the gradient calculate using a portion of the samples'''
    # Get index of first sample for the mini batch
    m = Y.shape[0]
    b = samples_list[index : index+MINI_SIZE] # Get indexes for mini batch
    gradient_vals = (1/MINI_SIZE)*(X[b].dot(T) - Y[b]).T.dot(X[b])

    # Update epoch global vars
    update_epoch(X, T, Y, MINI_SIZE, m)

    return (gradient_vals).T

def stoch_gradient(X, T, Y):
    ''' Returns the gradient calculated using a single random sample.'''
    m = Y.shape[0]
    i = [samples_list[index]] # Get next sample
    gradient_vals = (X[i].dot(T) - Y[i])*(X[i]).T

    # Update epoch global vars
    update_epoch(X, T, Y, 1, m)

    return gradient_vals

def sk_regressor(X, Y):
	'''
		Returns the sklearn model fitted

	    X is the features matrix
	    Y is the target array
	    Both need to be numpy arrays
	'''

	clf = linear_model.SGDRegressor(max_iter = ITER_LIMIT, tol = STEP_LIMIT, alpha = ALPHA)
	clf.fit(X,Y)
	return clf

def normal_equation(X, Y):
	'''
		Returns the analytical solution via normal equation.
		
		X is the features matrix
		Y is the target array
		Both need to be numpy arrays
	'''
	
	# Normal equation: step 1
	square = X.T.dot(X)
	
	# Check if matrix is invertible
	if np.linalg.det(square) == 0:
		print("Matrix not invertible! Cannot be solved by normal equation.")
		return None
	
	# Rest of equation
	theta = ((np.linalg.inv(square)).dot(X.T)).dot(Y)
	return theta

### global epoch variables ####
# References the current sample(s) used, ensuring that gradients
# such as stoch and minib iterate through all the samples instead
# of repeating samples in each iteration. Also provides a shuffled
# list if samples indexes for randomizations and lists for keeping
# epoch data.
samples_list = []
index = 0
new_epoch = 0
epochs_count = 0
epochs_info = [[],[]] # Time and Cost per Epoch 
start_time = 0

def boot_epoch_data(X, T, Y):
    global start_time, epochs_info, samples_list 
    start_time = time.time() # Set starting time
    epochs_info[0].append(cost(X, T, Y)) # Set starting cost
    epochs_info[1].append(0.0)             # Set starting time
    samples_list = list(range(Y.shape[0])) # Set and shuffle samples index
    shuffle(samples_list)

def update_epoch(X, T, Y, increment, bound):
    '''Update epochs global variables when a epoch is completed'''
    global epochs_info, epochs_count, index, samples_list, new_epoch 
    
    # Update samples count
    index += increment
    if index < bound : return
    
    # If epoch was completed
    index = 0             # Reset samples index
    epochs_info[0].append(cost(X, T, Y)) # Adds current epoch cost
    epochs_info[1].append(time.time() - start_time) # Adds time until epoch is done
    epochs_count += 1     # Count epoch
    shuffle(samples_list) # Reshuffle samples
    new_epoch = 0         # Lower new epoch flag

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
X,_,Y = numpy_and_bias(X,Y)

T = descent(X, T, Y, type='s')
print(T)
print("Cost =>",cost(X,T,Y))
print(epochs_count)
print(epochs_info[0][0],epochs_info[0][-1])