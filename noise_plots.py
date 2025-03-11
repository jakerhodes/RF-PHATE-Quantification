import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from rfphate import RFPHATE, dataprep

import embed
from joblib import Parallel, delayed

from rfphate import low_dimensional_group_separation
from rfphate import local_structure_preservation
from rfphate import global_structure_preservation
import pickle

def concat_noise(x, n_noise_vars = 100, distribution = 'uniform', random_state = 42):
    rng = np.random.default_rng(random_state)
    n, _ = np.shape(x)
    
    if distribution == 'uniform':
        noise_vars = rng.uniform(0, 1, (n, n_noise_vars))

    elif distribution == 'normal':
        noise_vars = rng.normal(0, 1, (n, n_noise_vars))

    else:
        raise ValueError('Distribution not recognized. Please choose either "uniform" or "normal".')

    x_noisy = np.concatenate((x, noise_vars), axis = 1)

    return x_noisy

datasets = ['audiology', 'balance_scale', 'breast_cancer', 'car', 'chess', 'crx',
            'diabetes', 'ecoli_5', 'flare1', 'glass', 'heart_disease',
            'heart_failure', 'hepatitis', 'ionosphere', 'iris', 'lymphography',
            'optdigits', 'parkinsons', 'seeds', 'segmentation', 'tic-tac-toe',
            'titanic', 'artificial_tree', 'waveform', 'wine', 'zoo']


unsupervised_methods = embed.unsupervised_methods
supervised_methods   = embed.supervised_methods
rf_methods           = embed.rf_methods

all_methods = unsupervised_methods + supervised_methods + rf_methods

def process_dataset(dataset):
    # Read in the data
    data = pd.read_csv('./data/' + dataset + '.csv', sep=',')
    x, y = dataprep(data)
    n, d = x.shape
    n_classes = len(y.unique())
    x_noisy = concat_noise(x, n_noise_vars=500, distribution='uniform', random_state=42)
    
    dataset_results = {}
    
    for method in all_methods:
        emb = method(x, y, random_state=42)
        emb_noisy = method(x_noisy, y, random_state=42)
        dataset_results[method.__name__] = emb
        dataset_results[method.__name__ + '_noisy'] = emb_noisy

        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].scatter(emb_noisy[:, 0], emb_noisy[:, 1], c=y, cmap = 'tab10', s=5)
        ax[0].set_xlabel('Dimension 1')
        ax[0].set_ylabel('Dimension 2')
        ax[0].set_title(dataset.upper() + ' + Noise')

        ax[1].scatter(emb[:, 0], emb[:, 1], c=y, cmap = 'tab10', s = 5)
        ax[1].set_xlabel('Dimension 1')
        ax[1].set_ylabel('Dimension 2')
        ax[1].set_title(dataset.upper())

        plt.tight_layout()
        plt.savefig('figures/' + dataset + '_' + method.__name__, bbox_inches='tight')

    return dataset, dataset_results

results = dict(Parallel(n_jobs=-1)(delayed(process_dataset)(dataset) for dataset in datasets))

with open('results.pkl', 'wb') as f:
    pickle.dump(results, f)