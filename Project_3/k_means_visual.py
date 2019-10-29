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
pca, Xpca = red.reduce_PCA(X, variance[0])

# K-Means
km, clusters = clus.k_means(Xpca, N_CLASSES, 'k-means++', 1000, 1e-4)

# # debscan
# DBSCAN(min_samples=10,)

# Create true_labelXcluster_label frequency matrix
count = np.zeros((N_CLASSES, N_CLASSES), dtype=int)
for i in range(N_CLASSES):
    for j in clusters[Y_true==i]:
        count[i,j] += 1

# Assing a label (class) for each cluster
pred = np.zeros(len(clusters), dtype=int)
while (True):
    x = count.argmax()
    i = x//N_CLASSES
    j = x%N_CLASSES
    if count[i,j] < 0 : break
    count[i,:] = count[:,j] = -1
    pred[clusters==j] = i

# Build confusion matrix
CM = np.zeros((N_CLASSES,N_CLASSES), dtype=int)
for i in range(N_CLASSES):
    mask = (Y_true == i) # Get elements from class i
    for j in range(N_CLASSES):
        aux = (pred[mask]==j) # Get elements from class i predicted as class j 
        CM[i,j] = aux.sum() # Count times class i was predicted as class j

vis.plot_confusion_matrix(CM, classes, "k-means")
