import umap.umap_ as umap

def SUMAP(x, y, **kwargs):
    return umap.UMAP(**kwargs).fit_transform(x, y = y)