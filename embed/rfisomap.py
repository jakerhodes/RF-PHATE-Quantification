from sklearn import manifold
from rfphate.rfgap import RFGAP
from sklearn.preprocessing import MinMaxScaler
import numpy as np


def RFISOMAP(x, y, prox_method = 'rfgap', random_state = None, n_jobs=-1, **kwargs):

    prox_op = RFGAP(y = y, prox_method = prox_method, 
                    random_state = random_state, **kwargs)
        

    prox_op.fit(x, y)

    K = np.array(prox_op.get_proximities().todense())

        

    scaler = MinMaxScaler()
    K = scaler.fit_transform(K)
    D = np.sqrt(1 - K)

    return manifold.Isomap(n_components=2, n_jobs=n_jobs, metric = 'precomputed', **kwargs).fit_transform(D)
