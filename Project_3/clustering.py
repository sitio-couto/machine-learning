from sklearn.cluster import KMeans, OPTICS

def k_means(X, n_clusters, init='k-means++', max_iter=300, tolerance=1e-4):
    km = KMeans(n_clusters, init, max_iter=max_iter, tol=tolerance)
    km.fit(X)
    
    return km, km.predict(X)
    
def optics(X, min_samples):
    opt = OPTICS(min_samples=min_samples)
    Y = opt.fit_predict(X)
    return opt, Y
