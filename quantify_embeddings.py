import pandas as pd
from joblib import Parallel, delayed

import rfphate
import embed
import random
import rfphate.dataprep as dataprep


def load_parse_csv(path, dataset_name):
    data = pd.read_csv(path + dataset_name + '.csv', sep = ',')
    x, y = rfphate.dataprep(data)
    return x, y

def quantify_embeddings(data, method, labels = None, random_state = 42, **kwargs):

    # Check to see of works with all models
    embedding = method(data, labels, random_state = random_state)
    
    LGS = rfphate.low_dimensional_group_separation(embedding, labels, random_state = random_state)
    LSP = rfphate.local_structure_preservation(data, labels, embedding, random_state = random_state, **kwargs)
    GSP = rfphate.global_structure_preservation(data, labels, embedding, random_state = random_state, **kwargs)
    model_diffs = rfphate.model_embedding_diff(data, labels, embedding, random_state = random_state, **kwargs)

    return LGS, LSP[0], GSP[0], model_diffs['knndiff'], model_diffs['rfdiff'], model_diffs['knn_scores_x'], model_diffs['rf_scores_x'], model_diffs['knn_scores_emb'], model_diffs['rf_scores_emb']
    

def load_and_quantify(path, dataset, method, random_state):

    random.seed(random_state)
    x, y = load_parse_csv(path, dataset)
    results = quantify_embeddings(x, method, y, n_repeats = 5)
    n, d = x.shape

    return {'iter': random_state, 'dataset': dataset,
            'n': n, 'd': d,
            'method': method.__name__, 'LGS': results[0],
            'LSP': results[1], 'GSP': results[2],
            'knndiff': results[3],
            'rfdiff': results[4],
            'knn_scores_x': results[5],
            'rf_scores_x': results[6],
            'knn_scores_emb': results[7],
            'rf_scores_emb': results[8]}


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

    n_random_states = len(random_states)

    for dataset_name in datasets:

        print(dataset_name)

        results = Parallel(n_jobs = -2)(delayed(load_and_quantify)('data/', dataset_name, method, random_states[i]) for i in range(n_random_states) for method in all_methods)
        results_df = pd.DataFrame(results)
        agg_results = results_df.groupby(['dataset', 'method'])[['LGS', 'LSP', 'GSP', 'knndiff', 'rfdiff', 'knn_scores_x', 'rf_scores_x', 'knn_scores_emb', 'rf_scores_emb']].agg(['mean', 'std'])
        agg_results.to_csv('results/' + dataset_name + '.csv')