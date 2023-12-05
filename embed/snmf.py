from methods import snmf

def SNMF(x, y, **kwargs):
    return snmf.SNMF(**kwargs).fit_transform(x, y)