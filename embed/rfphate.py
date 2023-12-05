import rfphate

def RFPHATE(data, y, **kwargs):
    return rfphate.RFPHATE(n_components = 2, y = y, **kwargs).fit_transform(data, y)