import numpy as np
from random import uniform, seed
from datetime import datetime

# Set seed based on current time
seed(datetime.now())

class Network:
    def __init__(self, model, e=1):
        '''Initializes coeficients tables with the network wheights
        
        Parameters: 
            model (list) : List with the amount of nodes per layer (without bias)
            e (float) : Range for the random coeficients initialization
        '''
        self.theta = []
        # Use model array to define coeficients matrices dimensions
        for (m,n) in zip(model[:-1], model[1:]):
            # Initializes thetas with values between -e and e (and adds bias <m+1>)
            table = [[uniform(-e,e) for j in range(m+1)] for i in range(n)]
            self.theta.append(np.array(table)) # Add matrix to model

    def forward(self, features, nodes=0):
        '''Execute the forward propagation using the defined thetas
        
        Parameters: 
            features (numpy.ndarray) : Column vector with input features (without bias)
            nodes (list) : if instanciated, saves the nodes activation values for every layer

        Returns:
            numpy.ndarray : Array with the propagated value for the output layer
        '''
        m = features.shape[1] # Get amount of samples to be propagated
        if isinstance(nodes, list) : nodes += [np.vstack([[1]*m, features])]

        for table in self.theta : 
            features = self.sigmoid(table.dot(np.vstack([[1]*m, features])))
            if isinstance(nodes, list) : nodes += [np.vstack([[1]*m, features])]

        return features

    def backprop(self, X, Y, l=0):
        layer = [] # For keeping the activation values
        theta = self.theta # Alias for the parameters
        m = Y.shape[1] # Amount of samples backpropagated
        delta = [np.zeros(i.shape) for i in theta] # For keeping the partial derivatives
        H = self.forward(X, nodes=layer) # Calculate hypotesis for every output node of every sample
        sigma = [np.zeros(i.shape) for i in layer[1:]] # For keeping the activation errors (except input layer)

        sigma[-1] = H - Y # Get output layer error
        
        # Back propagate error to hidden layers (ignoring bias from thetas and layers)
        for i in reversed(range(1, len(sigma))):
            sig_d = layer[i][1:,]*(1-layer[i][1:,]) # Remove bias from layers for backpropagation
            sigma[i-1] = (theta[i][:,1:].T).dot(sigma[i])*sig_d # Remove bias from thetas as well

        # Accumulate derivatives values for every theta (should be done by epochs)
        for i in range(len(delta)):
            regularization = np.hstack([theta[i][:,[1]]*0, theta[i][:,1:]*l])
            delta[i] = (delta[i] + sigma[i].dot(layer[i].T))/m + regularization
        
        return 

    def sigmoid(self, x):
        '''Calculates the sigmoid for the given value(s)

        Parameters:
            x (numpy.ndarray): matrix with the values with the function is applied

        Returns:
            numpy.ndarray : matrix with the trasnformed values
        '''
        return 1/(1 + np.exp(-x))

    def cost(self, X, Y, l=0):
        '''Calculates the current cost for the given samples (features and outputs)

        Parameters:
            X (numpy.ndarray): NxM matrix with N features and M samples
            Y (numpy.ndarray): KxM matrix with K output nodes and M samples
            l=0 (int): Regulrization parameter (0 disables regularization)

        Returns:
            float : Total cost for the current network and the given samples 
        '''
        m = Y.shape[1] # Get amount of 
        fun = lambda x : (x[:,1:]*x[:,1:]).sum() # Sum squared parameters without bias 
        H = self.forward(X) # Calculate hypotesis for every output node of every sample
        
        cost = -(Y*np.log(H) + (1-Y)*np.log(1-H)).sum()/m
        regularization = l*(sum(map(fun, self.theta))/(2*m))
        return cost + regularization


# train = np.load("dataset/train.npz")
# X, Y = train['xs'], train['ys']

import neural
import numpy as np
first = neural.Network([3,3,2,1])
feat = np.array([[1],[2],[1]])
Y = np.array([[1]])
first.backprop(feat, Y)
exit(1)