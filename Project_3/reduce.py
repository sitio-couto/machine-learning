from sklearn.decomposition import PCA

def reduce_PCA(X, variance):
    pca = PCA(variance)
    pca.fit(X)
    
    return pca, pca.transform(X)
