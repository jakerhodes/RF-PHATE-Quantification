from methods import esisomap

def ESISOMAP(x, y, **kwargs):
    return esisomap.ESIsomap(**kwargs).fit_transform(x, y)