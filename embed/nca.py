from sklearn.neighbors import NeighborhoodComponentsAnalysis

def NCA(x, y, **kwargs):
    return NeighborhoodComponentsAnalysis(n_components = 2, **kwargs).fit_transform(
        x, y
    )
