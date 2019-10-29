import numpy as np
from pandas import read_csv
from keras.utils import to_categorical
from const import *
import normalization as norm
import neural as nr
import reduce as red
import clustering as clus
import visualization as vis
import misc
import run


# Slice
m = 10000

data = read_csv('Dataset/fashion-mnist_train.csv')
Y_true = data['label'].to_numpy()[:m]
X = data.drop('label', 1).to_numpy()[:m,:]
classes = list(CLASS_NAMES.values())

# Normalization
choice = 2
stats = norm.get_stats(X, choice)
X = norm.normalize_data(X, stats, choice).astype('float32')

# Reduce PCA
variance = [.95]
pca, X = red.reduce_PCA(X, variance[0])

# Run Hierarchycal clustering
_, clusters = clus.agg_clustering(X, N_CLASSES, linkage='ward')

# Bind clusters to classes
Y_pred = clus.label_clusters(N_CLASSES, Y_true, clusters)

# Plot confusion matrix
CM = vis.build_confusion_matrix(N_CLASSES, Y_true, Y_pred)
vis.plot_confusion_matrix(CM, classes, "Agglomerate Clustering")
