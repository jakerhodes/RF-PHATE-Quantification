from sklearn import manifold


def ISOMAP(data, y = None, n_jobs=-1, **kwargs):
    return manifold.Isomap(n_components=2, n_jobs=n_jobs, **kwargs).fit_transform(data)