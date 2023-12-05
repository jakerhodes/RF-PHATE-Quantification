from sklearn import manifold


def MDS(data, y = None, n_jobs=-2, **kwargs):
    return manifold.MDS(n_components=2, n_jobs=n_jobs, **kwargs).fit_transform(data)