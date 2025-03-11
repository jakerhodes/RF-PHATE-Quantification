import warnings
from sklearn.manifold import Isomap
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from sklearn.decomposition import KernelPCA
from scipy.sparse.csgraph import connected_components
from scipy.sparse import issparse
import numpy as np
from scipy.spatial import distance_matrix
from scipy.sparse.csgraph import shortest_path
from sklearn.externals._packaging.version import parse as parse_version
from sklearn.utils.graph import _fix_connected_components
import scipy

class ESIsomap(Isomap):
    """
    Enhanced Supervised Isomap (ES-Isomap) for manifold learning with an adaptive distance metric.
    
    Parameters
    ----------
    n_neighbors : int, default=5
        Number of neighbors to consider for constructing the neighborhood graph.
    n_components : int, default=2
        Number of dimensions for the embedding.
    eigen_solver : str, {'auto', 'dense', 'arpack'}, default='auto'
        The eigenvalue decomposition strategy.
    tol : float, default=0
        Tolerance for convergence in eigen solver.
    max_iter : int, default=None
        Maximum number of iterations for solver.
    path_method : str, {'auto', 'FW', 'D'}, default='auto'
        Method to compute shortest paths.
    neighbors_algorithm : str, default='auto'
        Algorithm for nearest neighbor search.
    n_jobs : int, default=None
        Number of parallel jobs.
    p : int, default=2
        Power parameter for the Minkowski metric.
    sigma : float, default=1
        Scaling factor for ES-Isomap distance adjustment.
    d0 : float, default=0.5
        Offset parameter for ES-Isomap distance adjustment.
    """

    def __init__(
        self,
        *,
        n_neighbors=5,
        n_components=2,
        eigen_solver="auto",
        tol=0,
        max_iter=None,
        path_method="auto",
        neighbors_algorithm="auto",
        n_jobs=None,
        # metric="minkowski",
        p=2,
        sigma = 1, # For ES-Isomap
        d0 = .5,   # For ES-Isomap
        # metric_params=None,
    ):
    
        super(ESIsomap, self).__init__()

        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.eigen_solver = eigen_solver
        self.tol = tol
        self.max_iter = max_iter
        self.path_method = path_method
        self.neighbors_algorithm = neighbors_algorithm
        self.n_jobs = n_jobs
        # self.metric = metric
        self.p = p
        self.sigma = sigma
        self.d0 = d0
        # self.metric_params = metric_params


    def es_dist(self, X, y):

        dists = distance_matrix(X, X)

        a = 1 / (np.exp(-np.square(dists)) / self.sigma)
        D = np.full_like(dists, 0)

        for y_temp in np.unique(y):
            D[np.ix_(y_temp == y, y == y_temp)] = np.sqrt((a[np.ix_(y_temp == y, y_temp == y)] - 1)/ a[np.ix_(y_temp == y, y_temp == y)])
            D[np.ix_(y_temp == y, y_temp != y)] = np.sqrt(a[np.ix_(y_temp == y, y_temp != y)]) - self.d0

        return D


    def _fit_transform(self, X, y):

        D = self.es_dist(X, y)
        self.nbrs_ = NearestNeighbors(
            n_neighbors=self.n_neighbors,
            algorithm=self.neighbors_algorithm,
            metric='precomputed',
            p=self.p,
            # metric_params=self.metric_params,
            n_jobs=self.n_jobs,
        )
        self.nbrs_.fit(D)
        self.n_features_in_ = self.nbrs_.n_features_in_
        if hasattr(self.nbrs_, "feature_names_in_"):
            self.feature_names_in_ = self.nbrs_.feature_names_in_

        self.kernel_pca_ = KernelPCA(
            n_components=self.n_components,
            kernel="precomputed",
            eigen_solver=self.eigen_solver,
            tol=self.tol,
            max_iter=self.max_iter,
            n_jobs=self.n_jobs,
        )

        kng = kneighbors_graph(
            self.nbrs_,
            self.n_neighbors,
            metric='precomputed',
            p=self.p,
            # metric_params=self.metric_params,
            mode="distance",
            n_jobs=self.n_jobs,
        )

        # Compute the number of connected components, and connect the different
        # components to be able to compute a shortest path between all pairs
        # of samples in the graph.
        # Similar fix to cluster._agglomerative._fix_connectivity.
        n_connected_components, labels = connected_components(kng)
        if n_connected_components > 1:
            if self.metric == "precomputed" and issparse(D):
                raise RuntimeError(
                    "The number of connected components of the neighbors graph"
                    f" is {n_connected_components} > 1. The graph cannot be "
                    "completed with metric='precomputed', and Isomap cannot be"
                    "fitted. Increase the number of neighbors to avoid this "
                    "issue, or precompute the full distance matrix instead "
                    "of passing a sparse neighbors graph."
                )
            warnings.warn(
                "The number of connected components of the neighbors graph "
                f"is {n_connected_components} > 1. Completing the graph to fit"
                " Isomap might be slow. Increase the number of neighbors to "
                "avoid this issue.",
                stacklevel=2,
            )

            # use array validated by NearestNeighbors
            kng = _fix_connected_components(
                X=self.nbrs_._fit_X,
                graph=kng,
                n_connected_components=n_connected_components,
                component_labels=labels,
                mode="distance",
                metric=self.nbrs_.effective_metric_,
                **self.nbrs_.effective_metric_params_,
            )

        if parse_version(scipy.__version__) < parse_version("1.3.2"):
            # make identical samples have a nonzero distance, to account for
            # issues in old scipy Floyd-Warshall implementation.
            kng.data += 1e-15

        self.dist_matrix_ = shortest_path(kng, method=self.path_method, directed=False)

        G = self.dist_matrix_ ** 2
        G *= -0.5

        self.embedding_ = self.kernel_pca_.fit_transform(G, y)

    def fit_transform(self, X, y=None):
        """Fit the model from data in X and transform X.
        Parameters
        ----------
        X : {array-like, sparse graph, BallTree, KDTree}
            Training vector, where `n_samples` is the number of samples
            and `n_features` is the number of features.
        y : Ignored
            Not used, present for API consistency by convention.
        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
            X transformed in the new space.
        """
        self._fit_transform(X, y)
        return self.embedding_