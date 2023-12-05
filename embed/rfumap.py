import umap.umap_ as umap
from rfphate.rfgap import RFGAP
from sklearn.preprocessing import MinMaxScaler
import numpy as np


def RFUMAP(data, y, prox_method = 'rfgap', random_state = None, **kwargs):

    prox_op = RFGAP(y = y, prox_method = prox_method, 
                        random_state = random_state, **kwargs)
        

    prox_op.fit(data, y)

    K = np.array(prox_op.get_proximities().todense())


    scaler = MinMaxScaler()
    K = scaler.fit_transform(K)
    D = np.sqrt(1 - K)


    return umap.UMAP(metric = 'precomputed',
                     random_state = random_state, **kwargs).fit_transform(D)
