# MC886 - Machine Learning - UNICAMP
# Project 3 - Unsupervised Learning and Dimensionality Reduction
#
# Authors:
# Victor Ferreira Ferrari - RA 187890
# Vinicius Couto Espindola- RA 188115

from sklearn.metrics import confusion_matrix
import numpy as np
from pandas import read_csv
from keras.utils import to_categorical
from const import *
import normalization as norm
import neural as nr
import reduce as red
import seaborn as sns
import matplotlib.pyplot as plt
import clustering as clus
import visualization as vis
from scipy.stats import mode
from collections import Counter
import misc
import run

# Reading data
data = read_csv('Dataset/fashion-mnist_train.csv')
Y_true = data['label'].to_numpy()
X = data.drop('label', 1).to_numpy()
classes = list(CLASS_NAMES.values())

# Normalization
choice = 2
stats = norm.get_stats(X, choice)
X = norm.normalize_data(X, stats, choice).astype('float32')
Y = to_categorical(Y_true).astype('int8')

variance = [.95, .90, .85, .80, .75, .70]
for v in variance:
    # Applying PCA
    pca, Xpca = red.reduce_PCA(X, v)

    # K-Means
    km, clusters = clus.k_means(Xpca, N_CLASSES, 'k-means++', 1000, 1e-4)

    # Create prediction array and confusion matrix
    Y_pred = clus.label_clusters(N_CLASSES, Y_true, clusters)
    CM = vis.build_confusion_matrix(N_CLASSES, Y_true, Y_pred)
    # Plot confusion matrix
    vis.plot_confusion_matrix(CM, classes, f"k-means ({int(v*100)}% PCA)")
