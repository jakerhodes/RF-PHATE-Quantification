import os
import json
import numpy as np
import pandas as pd
import warnings

def process_json_files(folder_path):
    # Initialize lists to store results for each dataset
    local_scores = {}
    global_scores = {}
    models = []
    dataset_names = []

    # Suppress specific warnings related to mean of empty slices
    warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*Mean of empty slice.*')

    # Iterate through each JSON file in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            dataset_name = filename.split(".")[0]
            dataset_names.append(dataset_name)

            with open(os.path.join(folder_path, filename), 'r') as f:
                data = json.load(f)

            # Get the model names (keys from 'local_scores' and 'global_scores')
            if not models:
                models = list(data['local_scores'].keys())

            # Collect the local and global scores, handling NaN values for missing models
            local_scores[dataset_name] = {}
            global_scores[dataset_name] = {}
            for model in models:
                # Assign NaN if the model is missing in the current dataset's scores
                local_scores[dataset_name][model] = np.array(data['local_scores'].get(model, [np.nan] * len(data['local_scores'].get(model, []))))
                global_scores[dataset_name][model] = np.array(data['global_scores'].get(model, [np.nan] * len(data['global_scores'].get(model, []))))

    # Prepare the summary table
    local_summary = []
    global_summary = []
    
    for dataset_name in dataset_names:
        local_summary.append([f"{np.nanmean(local_scores[dataset_name][model]):.3f} ± {np.nanstd(local_scores[dataset_name][model]) if np.count_nonzero(~np.isnan(local_scores[dataset_name][model])) > 1 else np.nan:.3f}" for model in models])
        global_summary.append([f"{np.nanmean(global_scores[dataset_name][model]):.3f} ± {np.nanstd(global_scores[dataset_name][model]) if np.count_nonzero(~np.isnan(global_scores[dataset_name][model])) > 1 else np.nan:.3f}" for model in models])

    # Aggregated row (avg and std across all datasets)
    aggregated_local = []
    aggregated_global = []
    
    for model in models:
        # Check for non-empty data to avoid warnings
        local_data = [local_scores[dataset_name][model] for dataset_name in dataset_names if np.any(~np.isnan(local_scores[dataset_name][model]))]
        global_data = [global_scores[dataset_name][model] for dataset_name in dataset_names if np.any(~np.isnan(global_scores[dataset_name][model]))]
        
        # Handle empty data case
        if local_data:  # Ensure there is data to compute mean and std
            aggregated_local_mean = np.nanmean([np.nanmean(ld) for ld in local_data])
            aggregated_local_stdev = np.nanstd([np.nanstd(ld) for ld in local_data]) if np.count_nonzero(~np.isnan(np.concatenate(local_data))) > 1 else np.nan
        else:
            aggregated_local_mean = np.nan
            aggregated_local_stdev = np.nan
        
        if global_data:  # Ensure there is data to compute mean and std
            aggregated_global_mean = np.nanmean([np.nanmean(gd) for gd in global_data])
            aggregated_global_stdev = np.nanstd([np.nanstd(gd) for gd in global_data]) if np.count_nonzero(~np.isnan(np.concatenate(global_data))) > 1 else np.nan
        else:
            aggregated_global_mean = np.nan
            aggregated_global_stdev = np.nan
        
        # Format the aggregated row
        aggregated_local.append(f"{aggregated_local_mean:.3f} ± {aggregated_local_stdev:.3f}")
        aggregated_global.append(f"{aggregated_global_mean:.3f} ± {aggregated_global_stdev:.3f}")

    local_summary.append(aggregated_local)
    global_summary.append(aggregated_global)

    # Convert to pandas DataFrame for easier viewing
    columns = models
    
    local_df = pd.DataFrame(local_summary, columns=columns, index=dataset_names + ['Aggregated'])
    global_df = pd.DataFrame(global_summary, columns=columns, index=dataset_names + ['Aggregated'])

    # Calculate the average score for each method across all datasets
    local_avg_scores = np.array([np.nanmean([np.nanmean(local_scores[dataset_name][model]) for dataset_name in dataset_names if np.any(~np.isnan(local_scores[dataset_name][model]))]) for model in models])
    global_avg_scores = np.array([np.nanmean([np.nanmean(global_scores[dataset_name][model]) for dataset_name in dataset_names if np.any(~np.isnan(global_scores[dataset_name][model]))]) for model in models])

    # Get the order of models based on the average scores (descending order)
    local_sorted_models = [models[i] for i in np.argsort(local_avg_scores)[::-1]]
    global_sorted_models = [models[i] for i in np.argsort(global_avg_scores)[::-1]]

    # Reorder the dataframes based on sorted model order
    local_df = local_df[local_sorted_models]
    global_df = global_df[global_sorted_models]

    return local_df, global_df

folder_path = '/NOBACKUP/aumona/projects/rf-autoencoders/results/importance_preservation/'
local_df, global_df = process_json_files(folder_path)

print("Local Scores Table:")
print(local_df)
print("\nGlobal Scores Table:")
print(global_df)
