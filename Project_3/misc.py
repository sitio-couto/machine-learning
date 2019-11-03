import numpy as np

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
    
# Precision per class from confusion matrix.
def precision(confusion):
    diag = np.diagonal(confusion)
    return diag/confusion.sum(0)

# Recall per class from confusion matrix.
def recall(confusion):
    diag = np.diagonal(confusion)
    return diag/confusion.sum(1)

# F1 Score per class from precision and recall
def f1_score(prec, rec, bound=1e-8):
    return 2*prec*rec/(prec+rec+bound)
    

# Function to calculate metrics for evaluation
def get_metrics(target, predictions, classes):
    conf = confusion_matrix(target, predictions, classes)
    
    # Metrics
    acc = accuracy(conf)
    prec= precision(conf)
    rec = recall(conf)
    f1 = f1_score(prec, rec)
    avg_acc = (prec + rec)/2
    
    return {'accuracy':acc, 'norm_acc':avg_acc, 'precision':prec, 'recall':rec, 'f1':f1}, conf
