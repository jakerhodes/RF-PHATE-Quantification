from sklearn import decomposition

def KPCA(x, y = None, random_state = None, **kwargs):

    return decomposition.KernelPCA(n_components = 2, random_state = random_state, kernel = 'rbf', **kwargs).fit_transform(
        x
    )
