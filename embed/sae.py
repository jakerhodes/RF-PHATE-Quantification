from methods import sae

# TODO: Still to be tested
def SAE(x, y, random_state=None, **kwargs):
    model = sae.SAE(random_state = random_state, **kwargs)
    embedding = model.fit_transform(x, y)
    return embedding