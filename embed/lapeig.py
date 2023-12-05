from scipy.spatial import distance_matrix
import numpy as np
from sklearn import manifold


def LAPEIG(data, y = None, **kwargs):
    
    return manifold.SpectralEmbedding(n_components = 2, **kwargs).fit_transform(
        data
    )

