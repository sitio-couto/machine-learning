import numpy as np

# Data, for now, has to be a list or numpy matrix.
def normal_equation(data):
	
	# If the data is not a numpy array. 
	arr = np.array(data)
	
	# Y: Values. X: Features
	Y = arr[:,-1]
	X = arr[:, :-1]
	X = np.insert(X, 0, 1, axis=1)
	
	# Normal equation
	theta = ((np.linalg.inv(X.T.dot(X))).dot(X.T)).dot(Y)
	return theta
