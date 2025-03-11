from methods import esisomap

def ESISOMAP(x, y, random_state = None, **kwargs):
    return esisomap.ESIsomap(**kwargs).fit_transform(x, y)