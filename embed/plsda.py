from sklearn.cross_decomposition import PLSRegression

def PLSDA(x, y, **kwargs):
    return PLSRegression(n_components = 2, **kwargs).fit_transform(
        x, y
    )[0]
