import numpy as np
from sklearn.cluster import KMeans, OPTICS, AgglomerativeClustering

def k_means(X, n_clusters, init='k-means++', max_iter=300, tolerance=1e-4):
    km = KMeans(n_clusters, init, max_iter=max_iter, tol=tolerance)
    km.fit(X)
    
    return km, km.predict(X)
    
def optics(X, min_samples):
    opt = OPTICS(min_samples=min_samples)
    Y = opt.fit_predict(X)
    return opt, Y

def agg_clustering(X, n_clusters, linkage='ward'):
    aggclust = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage).fit(X)
    return aggclust, aggclust.labels_

def label_clusters(n_classes, Y_true, clusters):
    '''Binds each of the clusters labels to a class label creating a prediction array
    
        Parameters:
            n_classes (int): Amount of classes/clusters in the dataset/model
            clusters (array of int): cluster associated to each sample (sample i binds to cluster x[i])
            Y_true (array of int): class associated to each sample (sample i belongs to class x[i])

        Returns:
            Y_pred (array of int): contains the predicted class of sample i which is binded to the cluster clusters[i]
    '''

    # Create true_labelXcluster_label frequency matrix
    count = np.zeros((n_classes, n_classes), dtype=int)
    for i in range(n_classes):
        for j in clusters[Y_true==i]:
            count[i,j] += 1

    # Assing a label (class) for each cluster
    Y_pred = np.zeros(len(clusters), dtype=int)
    while (True):
        x = count.argmax()
        i = x//n_classes
        j = x%n_classes
        if count[i,j] < 0 : break
        count[i,:] = count[:,j] = -1
        Y_pred[clusters==j] = i

    return Y_pred