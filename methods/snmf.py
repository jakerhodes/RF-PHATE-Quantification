import numpy as np
from scipy.spatial import distance_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.base import BaseEstimator, TransformerMixin


class SNMF(BaseEstimator, TransformerMixin):

    def __init__(self, k = 2, lamb = 1, mu = 1, p = 5, sigma = 1, max_iters = 1000, random_state = None):

        self.k = k
        self.lamb = lamb
        self.mu = mu
        self.p = p
        self.max_iters = max_iters
        self.sigma = sigma
        self.random_state = random_state

        super(SNMF, self).__init__()

    def _gaussian_kernel(self, dist):
        return np.exp(-np.square(dist)/np.square(self.sigma))
        
    def fit(self, X, y):
        self._fit_transform(X, y)

    def _fit_transform(self, X, y):
        
        np.random.seed(self.random_state)
        n, d = np.shape(X)
        
        Shat = np.zeros((n, n))
        nbrs = NearestNeighbors(n_neighbors = self.p).fit(X)
        dists, nn = nbrs.kneighbors(X)
        

        for i in range(n):
            for j in range(self.p):
                j_idx = nn[i, j]
                Shat[i, j_idx] = self._gaussian_kernel(dists[i, j])

        D = np.ones((n, n))
        Sbar = np.zeros((n, n))
        
        U = np.random.uniform(size = (d, self.k))
        V = np.random.uniform(size = (self.k, n))  

        for i in range(n):
            for j in range(n):
                if i > j:
                    continue
                else:
                    if y[i] == y[j]:
                        D[i, j] = 0
                        D[j, i] = 0
                        Sbar[i, j] = 1
                        Sbar[j, i] = 1


        S = Shat + Sbar

        iters = 1
        while iters < self.max_iters:
            A = np.diag(np.sum(S, axis = 1))
            
            XVT = np.matmul(np.transpose(X), np.transpose(V))
            UVVT = np.matmul(U, np.matmul(V, np.transpose(V)))
            
            UTX = np.matmul(np.transpose(U), np.transpose(X))
            VS = 2 * self.mu * np.matmul(V, S)
            UTUV = np.matmul(np.matmul(np.transpose(U), U), V)
            VA = 2 * self.mu * np.matmul(V, A)
            VD = self.lamb * np.matmul(V, D)
            
            for i in range(d):
                for j in range(self.k):
                    U[i, j] = U[i, j] * XVT[i, j] / (UVVT[i, j] + 0.000001)
        
            
            for i in range(self.k):
                for j in range(n):
                    V[i, j] = V[i, j] * (UTX[i, j] + VS[i, j]) / (UTUV[i, j] + VA[i, j] + VD[i, j])

            iters = iters + 1

        self.VT = np.transpose(V)


    def fit_transform(self, X, y):
        self._fit_transform(X, y)
        return self.VT
