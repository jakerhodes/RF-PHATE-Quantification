from methods import ssnp

# TODO: Test this model
def SSNP(x, y = None, random_state = None, **kwargs):
    model = ssnp.SSNP(random_state = random_state, **kwargs)
    model.fit(x, y)
    embedding = model.transform(x)
    return embedding