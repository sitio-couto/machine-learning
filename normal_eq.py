import numpy as np

# Data, for now, has to be a list or numpy matrix.
def normal_equation(data):
	
	# If the data is not a numpy array. 
	arr = np.array(data)
	
	# Y: Target. X: Features/Examples
	Y = arr[:,-1]
	X = arr[:, :-1]
	X = np.insert(X, 0, 1, axis=1)
	
	# Normal equation: step 1
	square = X.T.dot(X)
	
	# Check if matrix is invertible
	if np.linalg.det(square) == 0:
		print("Matrix not invertible! Cannot be solved by normal equation.")
		return None
	
	# Rest of equation
	theta = ((np.linalg.inv(square)).dot(X.T)).dot(Y)
	return theta
