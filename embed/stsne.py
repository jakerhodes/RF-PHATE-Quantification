from scipy.spatial import distance_matrix
import numpy as np
from sklearn import manifold


def es_dist(X, y, sigma = None, d0 = 0.5):

    dists = distance_matrix(X, X)

    if sigma is None:
        sigma = np.mean(dists)

    a = 1 / (np.exp(-np.square(dists) / sigma))
    
    D = np.full_like(dists, 0)

    for y_temp in np.unique(y):

        D[np.ix_(y_temp == y, y == y_temp)] = np.sqrt((a[np.ix_(y_temp == y, y_temp == y)] - 1)/ a[np.ix_(y_temp == y, y_temp == y)])
        D[np.ix_(y_temp == y, y_temp != y)] = np.sqrt(a[np.ix_(y_temp == y, y_temp != y)]) - d0

    return D


def STSNE(x, y, sigma = None, d0 = 0.5, perplexity = 30, 
          random_state = None, **kwargs):

    D = es_dist(X = x, y = y, sigma = sigma, d0 = d0)
    
    return manifold.TSNE(n_components = 2, perplexity=perplexity, metric = 'precomputed',
                         init = 'random',  random_state = random_state, **kwargs).fit_transform(
        D
    )

