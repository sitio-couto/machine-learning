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

# Spliting sets
train = sample(range(60000), 50000)
valid = list(set(range(60000))-set(train))
Xv = X[valid,:]
Yv = Y[valid]
X = X[train,:]
Y = Y[train]

# Normalization
choice = 2
stats = norm.get_stats(X, choice)
X = norm.normalize_data(X, stats, choice).astype('float32')
stats = norm.get_stats(Xv, choice)
Xv = norm.normalize_data(Xv, stats, choice).astype('float32')

# Reduce PCA
print(f"Original: {X.shape[1]}")
variance = [.95, .90, .85, .80, .75, .70]
for v in variance:
    # Reduce features
    _, Xpca = red.reduce_PCA(X, v)
    _, Xvpca = red.reduce_PCA(Xv, v)
    print(f"Reduced to {Xpca.shape[1]}")
    # Run Hierarchycal clustering
    model, clusters = clus.k_means(Xpca, N_CLASSES, 'k-means++', 1000, 1e-4)
    # Bind clusters to classes
    Y_pred = clus.label_clusters(N_CLASSES, Y, clusters)
    Yv_pred = clus.label_clusters(N_CLASSES, Yv, model.fit_predict(Xvpca))
    # Get accuracies
    train_acc = vis.clustering_accuracy(N_CLASSES, Y, Y_pred)*100
    valid_acc = vis.clustering_accuracy(N_CLASSES, Yv, Yv_pred)*100
    print("Train: ", train_acc)
    print("Valid: ", valid_acc)
    # Get Confusion Matrix
    CM = vis.build_confusion_matrix(N_CLASSES, Y, Y_pred)
    np.save(f"cm_train_kmeans_{v}pca", CM)
    CM = vis.build_confusion_matrix(N_CLASSES, Yv, Yv_pred)
    np.save(f"cm_val_kmeans_{v}pca", CM)
