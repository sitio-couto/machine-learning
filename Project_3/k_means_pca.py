import numpy as np
from pandas import read_csv
from const import *
import normalization as norm
import reduce as red
import clustering as clus
import visualization as vis
from random import sample

# Reading data
data = read_csv('Dataset/fashion-mnist_train.csv')
Y = data['label'].to_numpy()
X = data.drop('label', 1).to_numpy()
classes = list(CLASS_NAMES.values())

# Normalization
choice = 2
stats = norm.get_stats(X, choice)
X = norm.normalize_data(X, stats, choice).astype('float32')

# Reduce PCA
print(f"Original: {X.shape[1]}")
variance = [.95, .90, .85, .80, .75, .70]
for v in variance:
    # Reduce features
    _, Xpca = red.reduce_PCA(X, v)
    print(f"Reduced to {Xpca.shape[1]}")
    
    # Run K-Means
    model, clusters = clus.k_means(Xpca, N_CLASSES, 'k-means++', 1000, 1e-4)
    vis.pca_plotting(Xpca, Y_pred, classes)
    
    # Testing
    dic, CM = clus.test_clusters(pred, Yl, N_CLASSES, Xpca)
    vis.plot_confusion_matrix(CM, CLASS_NAMES, 'K-Means')
    print(dic)
