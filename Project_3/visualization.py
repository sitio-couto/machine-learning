import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import itertools as it
import umap
import misc

def histogram(pred, amount=10, model='K-Means'):
    plt.hist(pred, bins='auto')
    plt.title("Amount of images per predicted class for " + model)
    plt.xticks(range(amount))
    #plt.grid()
    plt.show()

def pca_plotting(X, clust, classes):
    bidim = PCA(n_components=2)
    Xpca = bidim.fit_transform(X)

    for c,l in enumerate(classes):
        plt.scatter(Xpca[clust==c,0], Xpca[clust==c,1], label=l)
    
    plt.xlabel("First Component")
    plt.ylabel("Second Component")
    plt.legend()
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
