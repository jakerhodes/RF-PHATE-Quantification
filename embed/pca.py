from sklearn import decomposition

def PCA(x, y = None, **kwargs):
    return decomposition.PCA(n_components = 2, **kwargs).fit_transform(
        x
    )
