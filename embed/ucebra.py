import cebra

def UCEBRA(data, labels = None, **kwargs):
    cebra_model = cebra.CEBRA(output_dimension = 2, batch_size = 512, **kwargs)
    return cebra_model.fit_transform(data)