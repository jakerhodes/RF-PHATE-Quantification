import umap.umap_ as umap

def UMAP(data, y = None, random_state = None, **kwargs):
    return umap.UMAP(random_state = random_state, **kwargs).fit_transform(data)