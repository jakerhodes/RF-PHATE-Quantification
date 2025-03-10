from scipy.spatial import distance_matrix
import numpy as np
from sklearn import manifold


def es_dist(X, y, beta = None):

    dists = distance_matrix(X, X)

    if beta is None:
        beta = np.mean(dists)

    a = 1 / (np.exp(-np.square(dists) / beta))
    
    D = np.full_like(dists, 0)

    for y_temp in np.unique(y):

        D[np.ix_(y_temp == y, y == y_temp)] = np.sqrt((a[np.ix_(y_temp == y, y_temp == y)] - 1)/ a[np.ix_(y_temp == y, y_temp == y)])
        D[np.ix_(y_temp == y, y_temp != y)] = np.sqrt(a[np.ix_(y_temp == y, y_temp != y)])

    return D


def SLAPEIG(data, labels, beta = None, **kwargs):

    D = es_dist(X = data, y = labels, beta = beta)

    # K = 1 - D # Need to use heat kernel on NN graph here.
    
    return manifold.SpectralEmbedding(n_components = 2, affinity = 'precomputed_nearest_neighbors', **kwargs).fit_transform(
        D
    )

