import numpy as np

# Activation function
def softmax(x):
    x -= np.max(x)          # Numeric Stability
    x_exp = np.exp(x)
    return x_exp/x_exp.sum()
    
def prob(X, T):
    return softmax(X.dot(T))

def predict(X, T):
    y_scores = X.dot(T)
    return np.argmax(y_scores, axis=1)

# Cost function
def cost(Y, Y_probs):
    correct_probs = Y_probs[np.arange(Y.size), Y]
    return (-1 * log(correct_probs)).mean()

def cost_derivative(X, Y, Y_probs):
    Y_probs[np.arange(Y.size),Y] -= 1
    Y_probs /= Y.size
    return X.T.dot(Y_probs)

def log(x, bound=1e-16):
    return np.log(np.maximum(x,bound))

# Gradient Descent
def gd_step(X, Y, T, Y_prob, alpha):
    return T - alpha * cost_derivative(X, Y, Y_prob)

def gradient_descent(X, Y, X_v, Y_v, T, alpha=0.001, e_lim=1000):
    
    # First losses
    Y_prob = prob(X, T)
    Y_v_prob = prob(X_v, T)
    
    best_loss = cost(Y_v, Y_v_prob)
    best_T = T.copy()
    
    # Descent
    for i in range(e_lim):
        
        # New theta
        T = gd_step(X, Y, T, Y_prob, alpha)
        
        # New predictions and losses
        Y_prob = prob(X, T)
        loss = cost(Y, Y_prob)
        
        # Validation
        Y_v_prob = prob(X_v, T)
        v_loss = cost(Y_v, Y_v_prob)
        
        # Updating loss
        if v_loss < best_loss:
            best_loss = v_loss
            best_T = T.copy()
        
        # Early stopping for accuracy
        
        print(f"Epoch {i+1:04d}/{e_lim:04d}", f"loss: {loss:.4f} | val loss: {v_loss:.4f}")
        
    return best_T
