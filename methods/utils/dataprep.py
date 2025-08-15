import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def dataprep(data, label_col_idx=0, transform='normalize', global_transform=False, cat_to_numeric=True):
    """
    This method normalizes or standardizes all non-categorical variables in an array.
    All categorical variables are kept.
    Categorical variables as supposed to be ordered, so that we can transform them to numerical
    and standardize them the same way as continuous variables to be on the same scale.
    
    If transform = "normalize", categorical variables are scaled from 0 to 1. The highest value
    is assigned the value of 1, the lowest value is assigned the value of 0.
    If global_transform = True, the normalization is done globally (useful for image-like data), otherwise it is done feature-wise
                                                                                                    (useful for tabular data).
    """

    data = data.copy()
    categorical_cols = []

    for col in data.columns:
        if data[col].dtype == 'object' or data[col].dtype == 'int64':
            categorical_cols.append(col)

    if label_col_idx is not None:
        label = data.columns[label_col_idx]
        y = data.pop(label)
        x = data
    else:
        x = data
        y = None

    for col in x.columns:
        if col in categorical_cols and cat_to_numeric:
            x[col] = pd.Categorical(x[col]).codes

    # Ensure all features in x are floats
    x = x.astype(float)
    if not global_transform:
        if transform == 'standardize':
            for col in x.columns:
                std_dev = x[col].std()
                if std_dev == 0:  # Handle constant feature
                    x[col] = 0
                else:
                    x[col] = (x[col] - x[col].mean()) / std_dev
        elif transform == 'normalize':
            for col in x.columns:
                range_val = x[col].max() - x[col].min()
                if range_val == 0:  # Handle constant feature
                    x[col] = 0
                else:
                    x[col] = (x[col] - x[col].min()) / range_val
    else:
        if transform == 'standardize':
            global_mean = x.values.mean()
            global_std = x.values.std()
            denom = global_std if global_std != 0 else 1
            x = (x - global_mean) / denom
        elif transform == 'normalize':
            global_min = x.values.min()
            global_max = x.values.max()
            denom = global_max - global_min if global_max != global_min else 1
            x = (x - global_min) / denom                    
    if y is not None:
        y_encoded = LabelEncoder().fit_transform(y)

    return x, y_encoded
    