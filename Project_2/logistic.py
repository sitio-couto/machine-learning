import numpy as np

# Activation function
def softmax(x):
    x -= np.max(x)          # Numeric Stability
    x_exp = np.exp(x)
    return x_exp/x_exp.sum()
    
def prob(X, T):
    return softmax(X.dot(T))

# Cost function
def cost(Y, Y_probs):
    return (-1 * log(Y_probs[np.arange(Y.size), Y])).mean()

def cost_derivative(X, Y, Y_probs):
    
    Y_probs[:,Y] -= 1
    return X.T.dot(Y_probs)
    return np.log(np.maximum(x,bound))

def log(x, bound=1e-16):
    return np.log(np.maximum(x,bound))

# Gradient Descent
def gd_step(X, Y, T, Y_pred, alpha):
    return T - alpha * cost_derivative(X, Y, Y_pred)

def gradient_descent(X, Y, X_v, Y_v, T, alpha=0.001, e_lim=1000):
    
    # First predictions
    Y_prob = prob(X, T)
    Y_v_prob = prob(X_v, T)
    
    best_loss = cost(Y_v, Y_v_prob)
    best_T = T.copy()
    
    # Descent
    for i in range(e_lim):
        
        # New theta
        T = gd_step(X, Y, T, Y_prob, alpha)
        
        # New predictions and losses
        Y_pred = predict(X, T)
        loss = cost(Y, Y_prob)
        
        # Validation
        Y_v_pred = predict(X_v, T)
        v_loss = cost(Y_v, Y_v_prob)
        
        if v_loss < best_loss:
            best_loss = v_loss
            best_T = T.copy()
        
        print(f"Epoch {i+1:04d}/{e_lim:04d}", f"loss: {loss:.4f} | val loss: {v_loss:.4f}\n")
        
    return best_T
