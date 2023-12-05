from methods import eslle

def ESLLE(x, y, **kwargs):
    return eslle.ESLLE(**kwargs).fit_transform(x, y)