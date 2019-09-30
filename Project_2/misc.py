import numpy as np

def init_coefs(features, dim2, rand_seed=None):
    rand = np.random.RandomState(seed=rand_seed)
    return np.sqrt(2/(features + 1)) * rand.randn(features, dim2)

# Confusion matrix for the results.
def confusion_matrix(Y, Y_pred, classes):
    conf = np.zeros((classes,classes)).astype('int32')
    for i in range(Y.size):
        conf[Y[i], Y_pred[i]] += 1
    return conf

# Accuracy from confusion matrix. True/total
def accuracy(confusion):
    true_amount = confusion.trace()
    total = confusion.sum()
    return true_amount/total
    
