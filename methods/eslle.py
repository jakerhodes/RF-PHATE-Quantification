from matplotlib.pyplot import ylim
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import locally_linear_embedding
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from scipy.spatial import distance_matrix

class ESLLE(BaseEstimator, TransformerMixin, ):

    """
    Enhanced Supervised Locally Linear Embedding (ESLLE) implementation.
    
    This class extends Locally Linear Embedding (LLE) by incorporating supervised information 
    via a modified distance function that adjusts pairwise distances based on class labels.
    
    Parameters:
    -----------
    n_neighbors : int, default=5
        Number of neighbors to consider for each point.
    
    n_components : int, default=2
        Number of dimensions for the embedding.
    
    reg : float, default=1e-3
        Regularization parameter for the weight matrix in LLE.
    
    eigen_solver : {"auto", "arpack", "dense"}, default="auto"
        The eigenvalue decomposition strategy.
    
    tol : float, default=1e-6
        Tolerance for convergence in eigenvalue decomposition.
    
    max_iter : int, default=100
        Maximum number of iterations for the solver.
    
    method : {"standard", "hessian", "modified", "ltsa"}, default="standard"
        LLE algorithm type.
    
    hessian_tol : float, default=1e-4
        Tolerance parameter for Hessian LLE.
    
    modified_tol : float, default=1e-12
        Tolerance parameter for Modified LLE.
    
    neighbors_algorithm : {"auto", "ball_tree", "kd_tree", "brute"}, default="auto"
        Algorithm to use for nearest neighbor search.
    
    random_state : int, RandomState instance or None, default=None
        Determines random number generation for eigen solver.
    
    n_jobs : int or None, default=None
        Number of parallel jobs to run for nearest neighbor search.
    
    alpha : float, default=0.2
        Parameter controlling the influence of class separation in the modified distance function.
    
    beta : float or None, default=None
        Scaling factor for distance modification. If None, it is set to the mean pairwise distance.
    
    Methods:
    --------
    fit(X, y):
        Fits the ESLLE model to the given data.
    
    fit_transform(X, y):
        Fits the ESLLE model and returns the low-dimensional embedding.
    """
    def __init__(self,
        *,
        n_neighbors=5,
        n_components=2,
        reg=1e-3,
        eigen_solver="auto",
        tol=1e-6,
        max_iter=100,
        method="standard",
        hessian_tol=1e-4,
        modified_tol=1e-12,
        neighbors_algorithm="auto",
        random_state=None,
        n_jobs=None,
        alpha = 0.2,
        beta = None,
         **kwargs):

        super(ESLLE, self).__init__(**kwargs)
        
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.reg = reg
        self.eigen_solver = eigen_solver
        self.tol = tol
        self.max_iter = max_iter
        self.method = method
        self.hessian_tol = hessian_tol
        self.modified_tol = modified_tol
        self.random_state = random_state
        self.neighbors_algorithm = neighbors_algorithm
        self.n_jobs = n_jobs
        self.alpha = alpha
        self.beta = beta


        
    def _eslle_distance(self, X, y):

        dists = distance_matrix(X, X)

        if self.beta == None:
            self.beta = np.mean(dists)

        s_dists = np.full_like(dists, 1)

        for y_temp in np.unique(y):

            dists2 = np.square(dists)

            s_dists[np.ix_(y_temp == y, y != y_temp)] = np.sqrt(np.exp(dists2[np.ix_(y_temp == y, y_temp != y)] / self.beta)) - self.alpha
            s_dists[np.ix_(y_temp == y, y_temp == y)] = np.sqrt(1 - np.exp(-dists2[np.ix_(y_temp == y, y_temp == y)] / self.beta))

        return(s_dists)



    def fit(self, X, y):
        self._fit_transform(X, y)


    def _fit_transform(self, X, y):

        X = self._eslle_distance(X, y)

        self.nbrs_ = NearestNeighbors(
            n_neighbors=self.n_neighbors,
            algorithm=self.neighbors_algorithm,
            n_jobs=self.n_jobs,
            metric = 'precomputed'
        )

        # X = self._validate_data(X, dtype=float)
        self.nbrs_.fit(X)
        self.embedding_, self.reconstruction_error_ = locally_linear_embedding(
            X=self.nbrs_,
            n_neighbors=self.n_neighbors,
            n_components=self.n_components,
            eigen_solver=self.eigen_solver,
            tol=self.tol,
            max_iter=self.max_iter,
            method=self.method,
            hessian_tol=self.hessian_tol,
            modified_tol=self.modified_tol,
            random_state=self.random_state,
            reg=self.reg,
            n_jobs=self.n_jobs
        )

    def fit_transform(self, X, y):
        self._fit_transform(X, y)
        return(self.embedding_)


