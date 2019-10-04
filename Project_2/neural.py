import numpy as np
from random import uniform, seed, shuffle
from datetime import datetime
from time import time
from copy import deepcopy

# Set seed based on current time
seed(datetime.now())

class Meta:
    def __init__(self, T, m, batch_size, sampling=0):
        self.index = 0  # Saves the positon in the ramdom list
        self.iters = 0  # Counts the number of weights updates 
        self.bound = m  # Total amount of samples
        self.epochs_count = 0      # Amount of comleted epochs
        self.start_time = time()   # Training starting time
        self.sampling = sampling   # Numeber of iterations which samples are colected
        self.samples_list = list(range(m)) # Radomized samples for istocastic methods
        shuffle(self.samples_list)
        
        # Checks it the batch is an integer (n samples) or percentage and, if 
        # is a percentage, adjust to number of samples
        if not isinstance(batch_size, int) :
            self.batch_size = int(np.ceil(m*batch_size))
        else : 
            self.batch_size = batch_size
        
        # If analisys data will be kept, saves time and thetas
        if (self.sampling):
            self.coef = [deepcopy(T)]     # Keeps trained coeficients per epoch
            self.time = [0.0]   # Marks when a ecpoch was complete

    def update(self, T):
        '''Updates hyperparameters (epoch count, samples ramdomization)
        
            Parameters:
                T (list of np.ndarray) : Coeficients from the current epoch
                analisys (bool) : If true, keeps arguments for analisys
        '''
        # Update index (add samples used in iteration)
        self.iters += 1
        self.index += self.batch_size
        
        # If epoch completed
        if self.index >= self.bound :
            self.index = 0             # Reset samples index
            self.epochs_count += 1     # Count finished epoch
            shuffle(self.samples_list) # Reshuffle samples

        # Data for further analisys (Consumes time and memory)
        if (self.sampling and self.iters//self.sampling) :
            self.iters = 0
            self.time.append(time() - self.start_time) # Adds time until epoch is done
            self.coef.append(deepcopy(T)) # Adds current epoch cost

    def get_batch(self):
        '''Get samples indexes for the next batch'''
        return self.samples_list[self.index : self.index+self.batch_size]

    def __str__(self):
        out = f'<Meta Object at {hex(id(self))}>'
        out += f'Samples per Epoch: {self.bound}'
        out += f'Current samples index: {self.index}'
        out += f'Batch size: {self.batch_size}'
        out += f'Epochs complete so far: {self.epochs_count}'
        return out

###############################################################################

class Network:
    def __init__(self, model, f='lg', e=5, l=0, T=0, seed=0):
        '''Initializes coeficients tables with the network wheights
        
        Parameters: 
            model (list) : List with the amount of nodes per layer (without bias)
            f (String) : Identification for the function to be minimized
            e (float) : Range for the random coeficients initialization
            l (float) : Regularization parameter for the network (0 disables regularization)
            T (list of numpy.ndarray): If instanciated, uses T as initial thetas
        '''
        self.l = l
        self.f = f
        self.theta = []

        # Generates a random seed based on current time
        if not seed : int(divmod(time(), 1)[1])

        # If no preset, instanciate thetas and set random initial thetas
        for (n,m) in zip(model[1:],model[:-1]): 
            # Instanciate weights with values between -e and e
            rand = np.random.RandomState(seed=seed)
            self.theta.append((rand.rand(n,m+1).astype(np.float32)-0.5)*2*e)

    # Cost function
    def cost(self, X, Y):
        '''Calculates the current cost for the given samples (features and outputs)

        Parameters:
            X (numpy.ndarray): NxM matrix with N features and M samples
            Y (numpy.ndarray): KxM matrix with K output nodes and M samples

        Returns:
            float : Total cost for the current network and the given samples 
        '''
        reg = 0 # Regularization value (weight reduction)
        e = 10**-6 # Offset used to avoid log(0) (prevents NaNs)
        m = Y.shape[1]  # Get amount of samples
        fun = lambda x : (x[:,1:]*x[:,1:]).sum() # Sum squared parameters without bias 
        H = self.frag_forward(X, 10) # Get output layer activation values

        # Calculate cost function
        if self.f == 'lg': # Use logistic cost
            cost = -(Y*np.log(H+e) + (1-Y)*np.log((1+e)-H)).sum()/m
        elif self.f == 'sm': # Use softmax cost
            cost = (-Y * np.log(softmax(H)+e)).mean()

        # Calculate regularization, if parameter is set
        if self.l : reg = self.l*(sum(map(fun, self.theta))/(2*m))

        return cost + reg

    def cost_deriv(self, H, Y):
        '''Calculates the current cost for the given samples (features and outputs)

        Parameters:
            H (numpy.ndarray): NxM matrix with output layer activation values (N node X M samples)
            Y (numpy.ndarray): KxM matrix with K output nodes and M samples

        Returns:
            float : Total cost for the current network and the given samples 
        '''
        m = Y.shape[1]  # Get amount of samples

        if self.f =='lg': # Use logistic derivative
            return H - Y
        elif self.f == 'sm': # Use softmax derivative
            return softmax(H) - Y

    def forward(self, features, nodes=0):
        '''Execute the forward propagation using the defined thetas
        
        Parameters: 
            features (numpy.ndarray) : Column vector with input features (without bias)
            nodes (list) : if instanciated, saves the nodes activation values for every layer

        Returns:
            numpy.ndarray : Array with the propagated value for the output layer
        '''
        m = features.shape[1] # Get amount of samples to be propagated

        for table in self.theta :
            features = np.insert(features, 0, 1, axis=0)
            if isinstance(nodes, list) : nodes += [features] 
            features = sigmoid(table.dot(features))

        if isinstance(nodes, list) : nodes += [features]
        return features

    def backprop(self, X, Y):
        '''Execute gradient calculation for the given thetas and samples
        
        Parameters: 
            X (numpy.ndarray) : NxM matrix with M samples and each sample with N features
            Y (numpy.ndarray) : KxM matrix with M samples ana each samples with K output nodes

        Returns:
            list of numpy.ndarray : Gradients for each set of thetas in the network 
        '''
        l = self.l # Alias for the regularization parameter
        theta = self.theta # Alias for the parameters
        m = Y.shape[1] # Amount of samples to be backpropagated

        layer = [] # For keeping the activation values
        delta = [np.zeros(i.shape) for i in theta] # For keeping the partial derivatives
        H = self.forward(X, nodes=layer) # Calculate hypotesis for every output node of every sample
        sigma = [np.zeros(i.shape) for i in layer[1:]] # For keeping the activation errors (except input layer)
        reg = 0

        sigma[-1] = self.cost_deriv(H, Y)
        
        # Back propagate error to hidden layers (does not propagate to bias nodes)
        for i in reversed(range(1, len(sigma))):
            sig_d = layer[i][1:,:]*(1-layer[i][1:,:]) # Remove bias from layers for backpropagation
            sigma[i-1] = (theta[i][:,1:].T).dot(sigma[i])*sig_d # Remove bias from thetas as well

        # Accumulate derivatives values for every theta (does not update thetas)
        # - Biases are not regularized, so the biases weights are casted to zero
        for i in range(len(delta)):
            if l : reg = np.insert(theta[i][:,1:]*l, 0, 0, axis=1)
            delta[i] = (delta[i] + sigma[i].dot(layer[i].T))/m + reg
        
        return delta

    def train(self, X, Y, type='m', t_lim=7000, e_lim=100000, rate=0.01, mb_size=32, sampling=0):
        '''Trains the model until one of the given limits are reached

        Parameters:
            X (Float 2dArray): The coeficient matrix.
            Y (Float 2dArray): The results matrix.
            type (int): The choice of descent ('s'-stoch|'m'-mini|'b'-batch).

        Returns:
            Meta : Class containing the runtime info.
        '''
        # Initializes epochs metadata class
        batch_size = {'b':Y.shape[1],'m':mb_size,'s':1}.get(type) # Get number of samples
        data = Meta(self.theta, Y.shape[1], batch_size, sampling=sampling) # Saves hyperparameters and other info for analisys 

        # Starting descent
        while (time() - data.start_time) <= t_lim:
            # Getting new Thetas
            b = data.get_batch() # Get indexes for mini batch
            delta = self.backprop(X[:,b], Y[:,b])
            
            # Update coeficients
            for i in range(len(delta)): 
                self.theta[i] = self.theta[i] - rate*delta[i] 
            data.update(self.theta)

            # Check termination
            if data.epochs_count >= e_lim : 
                print("NOTE: Epochs limit for descent reached.")       
                return data
            
        print("NOTE: Time limit for descent exceded.")
        return data

    def frag_forward(self, X, parts):
        '''Wrapper for the forward propagation which splits the M samples in slices
           Prevents numpy memory spikes during large matrix multiplications           

        Parameters:
            X (Float 2dArray): NxM matrix with N input feratures and M samples
            type (int): The choice of descent ('s'-stoch|'m'-mini|'b'-batch).

        Returns:
            numpy.ndarray : Array with the propagated value for the output layer
        '''
        m = X.shape[1]
        size = int(np.ceil(m/parts)) # Get size of each batch
        out_layer = np.zeros((self.theta[-1].shape[0],m)) # Prealocate output layer
        batchs = [i*size for i in range(int(np.ceil(parts)+1))] # Slices indexes
        for (s,e) in zip(batchs[:-1], batchs[1:]): # Propagate for each slice
            out_layer[:,s:e] += self.forward(X[:,s:e])
        return out_layer

    def accuracy(self, X, Y, T=0):
        '''Caculates the cost for a set of samples in the current network
        
            Parameters:
                X (Float 2dArray): NxM matrix with N input features and M samples
                Y (Float 2dArray): NxM matrix with the expected M output layers
            
            Returns:
                float : Percentage of correct predictions for the M samples
        '''
        m = Y.shape[1] 
        H = self.frag_forward(X, 10)
        H = H.argmax(axis=0)
        Y = Y.argmax(axis=0)
        hits = (H==Y).sum()
        return hits*100/m

    def save(self, file_name):
        '''Function for saving the current network to a file'''
        save_list = [np.array([self.l,self.f])]
        save_list += self.theta
        np.savez(file_name, save_list)
        print(f"Model saved as {file_name}")
 
    def load(self, file_name):
        '''Function for loading a Network from a file'''
        obj = np.load(file_name)
        self.l = int(obj['arr_0'][0][0])
        self.f = str(obj['arr_0'][0][1])
        self.theta = []
        for T in obj['arr_0'][1:]:
            self.theta.append(T)
        print(f"Model {file_name} loaded")

    def __str__(self):
        funcs = {'lg':'Logistic', 'sm':'Softmax'}
        out = f'<Network Object at {hex(id(self))}>\n'
        out += f'Composed of {len(self.theta)+1} layers:\n'
        for i,n in enumerate(self.theta):
            out += f'   layer {i+1} - {n.shape[1]} nodes\n'
        out += f'   Out layer - {self.theta[-1].shape[0]} nodes\n'
        out += f'Cost function: {funcs[self.f]}\n'
        out += f'Regularization parameter: {self.l}\n'
        out += f'Amount of weights: {sum([x.size for x in self.theta])}\n'
        return out

###################################################
def softmax(x):
    x -= np.max(x, axis=0, keepdims=True)          # Numeric Stability
    x_exp = np.exp(x)
    return x_exp/x_exp.sum(axis=0, keepdims=True)

#####################################################

def sigmoid(x):
    '''Function for calculating the sigmoid and preventing overflow

        Parameters:
            x (float): Value for which the sigmoid will be calculated

        Returns:
            float : Sigmoid of x
    '''
    # The masks rounds values preventing np.exp to overflow
    x[x >  50] =  50.0
    x[x < -50] = -50.0
    return 1/(1 + np.exp(-x))

def cost(X, Y, T, l=0):
        '''Calculates the cost for the set of samples X and Y in the network T

        Parameters:
            X (numpy.ndarray): NxM matrix with N features and M samples
            Y (numpy.ndarray): KxM matrix with K output nodes and M samples
            T (list of numpy.ndarray): List with weights for each pair of layers
            l (float): Regularization parameter used to train the network

        Returns:
            float : Total cost for the current network and the given samples 
        '''
        e = 10**-6 # Offset used to avoid log(0) (prevents NaNs)
        m = Y.shape[1] # Get amount of 
        fun = lambda x : (x[:,1:]*x[:,1:]).sum() # Sum squared parameters without bias 
        
        # Forward propagation
        H = X
        for table in T : H = sigmoid(table.dot(np.vstack([[1]*m, H])))

        # Cost calculation
        cost = -(Y*np.log(H+e) + (1-Y)*np.log((1+e)-H)).sum()/m
        regularization = l*(sum(map(fun, T))/(2*m))
        return cost + regularization

def accuracy(X, Y, T):
    '''Caculates the cost for a set of samples in the current network
    
        Parameters:
            X (Float 2dArray): NxM matrix with N input features and M samples
            Y (Float 2dArray): NxM matrix with the expected M output layers
        
        Returns:
            float : Percentage of correct predictions for the M samples
    '''
    m = Y.shape[1] 
    H = frag_forward(X, 10, T)
    H = H.argmax(axis=0)
    Y = Y.argmax(axis=0)
    hits = (H==Y).sum()
    return hits*100/m

def forward(features, T, nodes=0):
    '''Execute the forward propagation using the defined thetas
    
    Parameters: 
        features (numpy.ndarray) : Column vector with input features (without bias)
        nodes (list) : if instanciated, saves the nodes activation values for every layer

    Returns:
        numpy.ndarray : Array with the propagated value for the output layer
    '''
    m = features.shape[1] # Get amount of samples to be propagated

    for table in T :
        features = np.insert(features, 0, 1, axis=0)
        if isinstance(nodes, list) : nodes += [features] 
        features = sigmoid(table.dot(features))

    if isinstance(nodes, list) : nodes += [features]
    return features

def frag_forward(X, parts, T):
    '''Wrapper for the forward propagation which splits the M samples in slices
        Prevents numpy memory spikes during large matrix multiplications           

    Parameters:
        X (Float 2dArray): NxM matrix with N input feratures and M samples
        type (int): The choice of descent ('s'-stoch|'m'-mini|'b'-batch).

    Returns:
        numpy.ndarray : Array with the propagated value for the output layer
    '''
    m = X.shape[1]
    size = int(np.ceil(m/parts)) # Get size of each batch
    out_layer = np.zeros((T[-1].shape[0],m)) # Prealocate output layer
    batchs = [i*size for i in range(int(np.ceil(parts)+1))] # Slices indexes
    for (s,e) in zip(batchs[:-1], batchs[1:]): # Propagate for each slice
        out_layer[:,s:e] += forward(X[:,s:e], T)
    return out_layer

# # Validation with "and" & "or" operations
# X = np.array([[0,0,1,1],[0,1,0,1]])

# Y_and = np.array([[0,0,0,1]])
# Y_or = np.array([[0,1,1,1]])
# Y_xor = np.array([[1,0,0,1]])
# Y_xnor = np.array([[0,1,1,0]])

# # Do not count bias when defining architecture
# and_op = Network([2,2,1], l=0)
# or_op = Network([2,2,1], l=0) 
# xor_op = Network([2,2,1], l=0) 
# xnor_op = Network([2,2,1], l=0) 

# # If does not converge, change hyperparameters
# Ys = [Y_and,Y_or,Y_xor,Y_xnor]
# nets = [and_op,or_op,xor_op,xnor_op]
# t = 'm'
# for i,name in enumerate(['AND', 'OR', 'XOR', 'XNOR']):
#     print(f'--({name})-------------------')
#     print("Initial cost:", nets[i].cost(X, Ys[i]))
#     print(np.round(nets[i].forward(X)))
#     nets[i].train(X, Ys[i], type=t, t_lim=10, e_lim=20000, mb_size=3)
#     print("Trained cost:", cost(X, Ys[i], nets[i].theta))
#     print(X)
#     print(np.round(nets[i].forward(X)))
