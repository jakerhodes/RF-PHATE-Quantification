
import graphtools
import numpy as np
from scipy import sparse
from rfphate.rfgap import RFGAP #TODO: Check this
from sklearn.preprocessing import MinMaxScaler

def RFDM(data, labels, prox_method = 'rfgap', random_state = None, t=1, knn=5, decay=40, **kwargs):
    # symmetric affinity matrix


    prox_op = RFGAP(y = labels, prox_method = prox_method, 
                        random_state = random_state, **kwargs)
        
    prox_op.fit(data, labels)

    K = np.array(prox_op.get_proximities().todense())

    scaler = MinMaxScaler()
    K = scaler.fit_transform(K)

    # degrees
    diff_deg = np.array(np.sum(K, axis=1)).flatten()
    # negative sqrt
    diff_deg = np.power(diff_deg, -1 / 2)
    # put into square matrix
    diff_deg = sparse.spdiags([diff_deg], diags=0, m=K.shape[0], n=K.shape[0])
    # conjugate
    K = sparse.csr_matrix(K)
    diff_aff = diff_deg.dot(K).dot(diff_deg)
    # symmetrize to remove numerical error
    diff_aff = (diff_aff + diff_aff.T) / 2
    # svd
    U, S, _ = sparse.linalg.svds(diff_aff, k=3)
    # sort by smallest eigenvector
    s_idx = np.argsort(S)[::-1]
    U, S = U[:, s_idx], S[s_idx]
    # get first eigenvector
    u1 = U[:, 0][:, None]
    # ensure non-zero
    zero_idx = np.abs(u1) <= np.finfo(float).eps
    u1[zero_idx] = (np.sign(u1[zero_idx]) * np.finfo(float).eps).reshape(-1)
    # normalize by first eigenvector
    U = U / u1
    # drop first eigenvector
    U, S = U[:, 1:], S[1:]
    # power eigenvalues
    S = np.power(S, t)
    # weight U by eigenvalues
    dm = U.dot(np.diagflat(S))
    return dm