from sklearn import manifold


def TSNE(data, y = None, perplexity=30, random_state = None, **kwargs):
    return manifold.TSNE(n_components=2, perplexity=perplexity,
                         random_state = random_state, **kwargs).fit_transform(
        data
    )