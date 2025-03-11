from sklearn.cross_decomposition import PLSRegression

def PLSDA(x, y, random_state = None, **kwargs):
    return PLSRegression(n_components = 2, **kwargs).fit_transform(
        x, y
    )[0]
