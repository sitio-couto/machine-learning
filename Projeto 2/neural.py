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

    def forward(self, features):
        '''Execute the forward propagation using the defined thetas
        
        Parameters: 
            features (numpy.ndarray) : Column vector with input features (without bias)

        Returns:
            numpy.ndarray : Array with the propagated value for the output layer
        '''
        m = features.shape[1] # Get amount of samples to be propagated

        for table in self.theta : 
            features = self.sigmoid(table.dot(np.vstack([[1]*m, features])))
        return features

    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))

    def cost(self, X, Y):
        pass


train = np.load("dataset/train.npz")
X, Y = train['xs'], train['ys']
