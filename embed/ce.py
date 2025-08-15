from methods import ce

# TODO: Still to be tested
def CE(x, y = None, random_state = None, **kwargs):
    model = ce.CE(random_state = random_state, **kwargs)
    model.fit(x, y)
    embedding = model.transform(x)
    return embedding