from sklearn import decomposition
from rfphate.rfgap import RFGAP
from sklearn.preprocessing import MinMaxScaler
import numpy as np


def RFKPCA(data, labels, prox_method = 'rfgap', random_state = None, **kwargs):

    prox_op = RFGAP(y = labels, prox_method = prox_method, 
                    random_state = random_state, **kwargs)
       

    prox_op.fit(data, labels)

    K = prox_op.get_proximities().toarray()

    scaler = MinMaxScaler()
    K = scaler.fit_transform(K)

    return decomposition.KernelPCA(n_components = 2, random_state = random_state, kernel = 'precomputed', **kwargs).fit_transform(
        K
    )
