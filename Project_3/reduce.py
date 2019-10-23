from sklearn.decomposition import PCA

def reduce_PCA(X, comp):
    pca = PCA(n_components=comp)
    pca.fit(X)
    
    return pca, pca.transform(X)
