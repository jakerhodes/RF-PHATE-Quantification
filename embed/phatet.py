import rfphate

# TODO: Add PageRankPHATE to init file in rfphate
def PHATET(data, y = None, verbose=False, n_jobs=-1, **kwargs):
    return rfphate.PageRankPHATE(verbose=verbose, n_jobs = n_jobs, **kwargs).fit_transform(data)