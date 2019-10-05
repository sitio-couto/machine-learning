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
    
    T, history = run_logistic(X, X_v, Y, Y_v, alpha, n_epochs, len(classes))

    # Learning curve
    vis.learning_with_history(history)

    # Testing
    print("Training Metrics:")
    test_logistic(X, Y, T, classes)
    
    print("Validation Metrics:")
    test_logistic(X_v, Y_v, T, classes)
    
def run_logistic(X, X_v, Y, Y_v, alpha, n_epochs, n_classes):
    '''
        Runs multinomial logistic regression.
        Uses Xavier Random Initialization of coefficients.
        Gradient descent: softmax function.
    '''
    
    T = misc.init_coefs(X.shape[1], n_classes, 57).astype('float32')

    print("Regression:")
    T, hist = lr.gradient_descent(X, Y, X_v, Y_v, T, alpha, n_epochs)
    return T, hist

def test_logistic(X, Y, T, classes):
    '''
        Tests logistic regression with implemented metrics.
        Accuracy, normalized accuracy, precision, recall, f1-score.
    '''
    
    pred = lr.predict(X, T)
    met, conf = misc.get_metrics(Y, pred, len(classes)) 
    vis.plot_confusion_matrix(conf, classes, model = 'Multinomial Logistic Regression')
    
    np.set_printoptions(precision=4)
    print(f'Accuracy: {met["accuracy"]:.4f}')
    print(f'Normalized Accuracy: {met["norm_acc"].mean():.4f}')
    print(f'Precision per class: {met["precision"]} (avg. precision: {met["precision"].mean():.4f})')
    print(f'Recall per class: {met["recall"]} (avg. recall: {met["recall"].mean():.4f})')
    print(f'F1-Score per class: {met["f1"]} (avg. f1-score: {met["f1"].mean():.4f})')
    print()

#### NEURAL NETWORK FUNCTIONS ####
def neural_network(X, Xv, Y, Yv, classes):
    '''
        Wrapper for neural network
    '''
    Xn, Yn, model = run_neural_network(X, Xv, Y, Yv)
    
    print("Training Metrics:")
    test_neural_network(Xn, Y, model, classes)
    
    print("Validation Metrics:")
    test_neural_network(Xv.T, Yv, model, classes)

def run_neural_network(X, Xv, Y, Yv):
    '''
        Runs fully-connected neural network.
    '''
    
    print("Starting neural network section")
    
    # Adjusting input matrices (assumes X is normalized)
    Xn = X.T
    Yn = norm.out_layers(Y)
    Xvn = Xv.T
    Yvn =norm.out_layers(Yv)
    print("Handled input")
    
    # Builds network object
    feat, out = Xn.shape[0], Yn.shape[0]
    h1, h2 = 256, 128
    arc = [feat, h1, h2, out]
    model = nr.Network(arc, f="sg", seed=23, reg_lambda=0.002)
    print('Created model')
    print("Initial Accuracy:", model.accuracy(Xn, Yn))
    
    # Train model
    batch_size = 1024
    data = model.train(Xn, Yn, Xvn, Yvn, type='m', mb_size=batch_size, e_lim=500, t_lim=3600, rate=0.001, opt='adam', betas=(0.9,0.999))
    
    return Xn, Yn, model

def test_neural_network(X, Y, model, classes):
    '''
        Function to test neural network.
    '''
    pred = model.predict(X)
    met, conf = misc.get_metrics(Y, pred, len(classes)) 
    vis.plot_confusion_matrix(conf, classes, model='Neural Network')
    
    np.set_printoptions(precision=4)
    print(f'Accuracy: {met["accuracy"]:.4f}')
    print(f'Normalized Accuracy: {met["norm_acc"].mean():.4f}')
    print(f'Precision per class: {met["precision"]} (avg. precision: {met["precision"].mean():.4f})')
    print(f'Recall per class: {met["recall"]} (avg. recall: {met["recall"].mean():.4f})')
    print(f'F1-Score per class: {met["f1"]} (avg. f1-score: {met["f1"].mean():.4f})')
    print()
    
def visualize_data(X, Y, Xv, Yv):
    X = X.T
    Y = norm.out_layers(Y)
    Xv = Xv.T
    Yv = norm.out_layers(Yv)
    # visu.sigmoid_vs_softmax(X, Y, Xv, Yv, n=(0,3073))
    # visu.comparing_models(X, Y, Xv, Yv, n=(0,3073))
    #visu.overfitting(X, Y, Xv, Yv)
    vis.optimization_cost(X, Y, Xv, Yv)
