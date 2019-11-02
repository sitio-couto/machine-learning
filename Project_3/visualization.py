import matplotlib.pyplot as plt
import numpy as np
import itertools as it
import umap
import misc

def build_confusion_matrix(n_classes, Y_true, Y_pred):
# Build confusion matrix
    CM = np.zeros((n_classes,n_classes), dtype=int)
    for i in range(n_classes):
        mask = (Y_true == i) # Get elements from class i
        for j in range(n_classes):
            aux = (Y_pred[mask]==j) # Get elements from class i predicted as class j 
            CM[i,j] = aux.sum() # Count times class i was predicted as class j

    return CM

def clustering_accuracy(n_classes, Y_true, Y_pred):
    CM = build_confusion_matrix(n_classes, Y_true, Y_pred)
    return misc.accuracy(CM)

def histogram(pred, amount=10, model='K-Means'):
    plt.hist(pred, bins='auto')
    plt.title("Amount of images per predicted class for " + model)
    plt.xticks(range(amount))
    #plt.grid()
    plt.show()

def plot_confusion_matrix(confusion, classes, model):
    '''
        Plots an already created confusion matrix for a generic amount of classes.
    '''
    
    fig, ax = plt.subplots(1)

    #Bounding box
    ax.spines['bottom'].set_color('black')
    ax.spines['top'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['right'].set_color('black')

    plt.title('Confusion Matrix for ' + model)

    #Ticks
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thr = confusion.max()/2
    for i, j in it.product(range(confusion.shape[0]), range(confusion.shape[1])):
        plt.text(j, i, confusion[i, j],
            horizontalalignment='center',
            color='white' if confusion[i, j] > thr else 'black')

    plt.grid(False)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

    plt.imshow(confusion, interpolation='nearest', cmap='Blues')
    plt.show()

    return fig, ax
