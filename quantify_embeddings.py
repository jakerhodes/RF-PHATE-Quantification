import pandas as pd
from joblib import Parallel, delayed
import warnings
warnings.simplefilter('ignore')

import rfphate
import embed
import random
import rfphate.dataprep as dataprep


def load_parse_csv(path, dataset_name):
    data = pd.read_csv(path + dataset_name + '.csv', sep = ',')
    x, y = dataprep(data)
    return x, y

def quantify_embeddings(data, method, labels = None, random_state = None, **kwargs):

    # Some methods do not have a random state
    try:
        embedding = method(data, y = labels, random_state = random_state)

    except:
        embedding = method(data, y = labels)
    
    LGS = rfphate.low_dimensional_group_separation(embedding, labels)
    LSP = rfphate.local_structure_preservation(data, labels, embedding, 
                                            random_state = random_state, **kwargs)
    
    GSP = rfphate.global_structure_preservation(data, labels, embedding,
                                                random_state = random_state, **kwargs)

    return LGS, LSP[0], GSP[0]
    

def load_and_quantify(path, dataset, method, random_state):
    x, y = load_parse_csv(path, dataset)
    results = quantify_embeddings(x, method, y, random_state, n_repeats = 30)
    return {'iter': random_state, 'dataset': dataset, 'method': method.__name__, 'LGS': results[0], 'LSP': results[1], 'GSP': results[2]}



if __name__ == '__main__':

    unsupervised_methods = embed.unsupervised_methods
    supervised_methods   = embed.supervised_methods
    rf_methods           = embed.rf_methods

    all_methods = unsupervised_methods + supervised_methods + rf_methods

    datasets = ['audiology', 'balance_scale', 'breast_cancer', 'car', 'chess', 'crx',
                'diabetes', 'ecoli_5', 'flare1', 'glass', 'heart_disease',
                'heart_failure', 'hepatitis', 'ionosphere', 'iris', 'lymphography',
                'optdigits', 'parkinsons', 'seeds', 'segmentation', 'tic-tac-toe',
                'titanic', 'artificial_tree', 'waveform', 'wine', 'zoo']

    # Pre-defined random states and call these in the main function
    random_states = [9923, 17654, 3456, 11234, 789, 15678,
                     4321, 19087, 5432, 8765, 12345, 9876,
                     543, 16543, 7890, 2345, 15000, 5678,
                     876, 19876]


    for dataset_name in datasets:

        print(dataset_name)

        results = Parallel(n_jobs = 4)(delayed(load_and_quantify)('data/', dataset_name, method, random_states[i]) for i in range(2) for method in all_methods)
        results_df = pd.DataFrame(results)
        agg_results = results_df.groupby(['dataset', 'method'])[['LGS', 'LSP', 'GSP']].agg(['mean', 'std'])
        agg_results.to_csv('results/' + dataset_name + '.csv')