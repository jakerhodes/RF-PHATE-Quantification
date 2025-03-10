from sklearn import manifold
from rfphate.rfgap import RFGAP
from sklearn.preprocessing import MinMaxScaler
import numpy as np


def RFLAPEIG(data, labels, prox_method = 'rfgap', random_state = None, **kwargs):

    prox_op = RFGAP(y = labels, prox_method = prox_method, 
                    random_state = random_state, **kwargs)
        

    prox_op.fit(data, labels)

    K = np.array(prox_op.get_proximities().todense())


    scaler = MinMaxScaler()
    K = scaler.fit_transform(K)

    return manifold.SpectralEmbedding(n_components=2, affinity = 'precomputed', **kwargs).fit_transform(
        K
    )
