from sklearn import manifold


def LLE(data, y = None, n_jobs=-1, **kwargs):
    return manifold.LocallyLinearEmbedding(
        n_components=2, **kwargs, n_jobs=n_jobs
    ).fit_transform(data)