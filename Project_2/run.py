import numpy as np
import visualization as vis
import normalization as norm
import logistic as lr
import neural as nr
import misc

#### MULTINOMIAL LOG. REGRESSION FUNCTIONS ####
def logistic(X, X_v, Y, Y_v, alpha, n_epochs, classes):
    '''
        Wrapper for logistic regression
    '''
    
    X = np.insert(X, 0, 1, axis=1)
    X_v = np.insert(X_v, 0, 1, axis=1)
    print("Bias Added")
    
    T = run_logistic(X, X_v, Y, Y_v, alpha, n_epochs, classes)

    # Testing
    print("\nTraining Metrics:")
    test_logistic(X, Y, T, classes)
    
    print("Validation Metrics:")
    test_logistic(X_v, Y_v, T, classes)
    
def run_logistic(X, X_v, Y, Y_v, alpha, n_epochs, classes):
    '''
        Runs multinomial logistic regression.
        Uses Xavier Random Initialization of coefficients.
        Gradient descent: softmax function.
    '''
    
    T = misc.init_coefs(X.shape[1], classes, 57).astype('float32')

    print("Regression:")
    T = lr.gradient_descent(X, Y, X_v, Y_v, T, alpha, n_epochs)
    return T

def test_logistic(X, Y, T, classes):
    '''
        Tests logistic regression with implemented metrics.
        Accuracy, normalized accuracy, precision, recall, f1-score.
    '''
    
    pred = lr.predict(X, T)
    met = misc.get_metrics(Y, pred, classes) 
    np.set_printoptions(precision=4)   
    print(f'Accuracy: {met["accuracy"]:.4f}')
    print(f'Normalized Accuracy: {met["norm_acc"].mean():.4f}')
    print(f'Precision per class: {met["precision"]} (avg. precision: {met["precision"].mean():.4f})')
    print(f'Recall per class: {met["recall"]} (avg. recall: {met["recall"].mean():.4f})')
    print(f'F1-Score per class: {met["f1"]} (avg. f1-score: {met["f1"].mean():.4f})')
    print()

#### NEURAL NETWORK FUNCTIONS ####
def neural_network(X,Y):
    '''
        Wrapper for neural network
    '''
    Xn, Yn, model = run_neural_network(X, Y)
    test_neural_network(Xn, Yn, model)

def run_neural_network(X, Y):
    '''
        Runs fully-connected neural network.
    '''
    
    print("Starting neural network section")
    
    # Adjusting input matrices (assumes X is normalized)
    Xn = X.T
    Yn = norm.out_layers(Y)
    print("Handled input")
    
    # Builds network object
    feat = Xn.shape[0]
    out = Yn.shape[0]
    batch_size = int(np.ceil(Xn.shape[1]*0.1))
    model = nr.Network([feat,feat,out])
    print('Created model')
    print("Initial Accuracy:", model.accuracy(Xn, Yn))
    
    # Train model
    data = model.train(Xn, Yn, type='m', mb_size=batch_size)
    
    return Xn, Yn, model

def test_neural_network(Xn, Yn, model):
    '''
        Function to test neural network.
    '''
    print("Trained Accuracy:", model.accuracy(Xn, Yn))

    # Neural Network descent visualization
    vis.learning_curves(Xn, Yn, m=80000)


