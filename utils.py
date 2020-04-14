import numpy as np
import pandas as pd

# Reads the csv files and normalizes it if needed
def read_csv(csv_filename, target="y", norm=False):
    df = pd.read_csv(csv_filename, delimiter=",", dtype={target: str})
    target2 = {target: i for i, target in enumerate(sorted(list(set(df[target].values))))}
    X = df.drop([target], axis=1).values
    y = np.vectorize(lambda x: target2[x])(df[target].values)
    n_classes = len(target2.keys())
    if norm:
        X = (X - X.mean(axis=0)) / X.std(axis=0)
    return X, y, n_classes

# folds the data and rturns the index of each fold
def crossval_folds(N, n_folds, seed=1):
    np.random.seed(seed)
    permuted_index = np.random.permutation(N)
    fold_size = int(N/n_folds)
    fold_index = []
    for i in range(n_folds):
        start = i*fold_size
        end = min([(i+1)*fold_size, N])
        fold_index.append(permuted_index[start:end])
    return fold_index

def dotprod(a, b):
    return sum([a_ * b_ for (a_, b_) in zip(a, b)])
