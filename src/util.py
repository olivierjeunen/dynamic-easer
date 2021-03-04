import functools
import numpy as np
import operator
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

import time

from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timedelta
from joblib import Parallel, delayed
from scipy.linalg import eigh, inv
from scipy.linalg.interpolative import estimate_rank # pylint: disable=no-name-in-module
from scipy.sparse import coo_matrix, csr_matrix, identity, lil_matrix
from scipy.sparse.linalg import eigsh
from scipy.stats import sem
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

def compute_k_core(clicks, user_col='session', item_col='item', user_k=5, item_k=5, i=1):
    '''
        Approximate the k-core for the 'clicks' dataframe.
        i.e. alternatingly drop users and items that appear less than 'k' times, for 'i' iterations.
    '''
    def drop_below_k(df, col, k):
        # Count occurence of values in `col`
        c = df[col].value_counts().reset_index().rename(columns = {'index': col, col: 'count'})
        # Only keep those that occur often enough
        c = c.loc[c['count'] >= k]
        # Join and return
        return df.merge(c, on = col, how = 'right').drop('count', axis = 1)

    for _ in range(i):
        clicks = drop_below_k(clicks, user_col, user_k)
        clicks = drop_below_k(clicks, item_col, item_k)
    
    return clicks

def print_summary(df, cols):
    '''
        Print number of unique values for discrete identifiers (users, items, ...)
    '''
    print(f"{len(df):10} unique user-item pairs...")
    for col in cols:
        print(f"{df[col].nunique():10} unique {col}s...")
    for col in cols:
        print(f"{df[col].value_counts().reset_index().rename(columns = {'index': col, col: 'count'})['count'].mean()} mean occurrences for {col}s...")
    

def encode_integer_id(col):
    '''
        Encode discrete values with unique integer identifiers
    '''
    return LabelEncoder().fit_transform(col)

def generate_csr(df, shape, user_col='session', item_col='item'):
    '''
        Encode user-item pairs into a Compressed Sparse Row (CSR) Matrix
    '''
    data = np.ones(len(df))
    rows, cols = df[user_col].values, df[item_col].values
    return csr_matrix((data, (rows, cols)), shape = shape)


def compute_gramian(X):
    '''
        Compute Gramian for user-item matrix X
    '''
    G = X.T @ X
    return G 

def add_diagonal(G, l2):
    '''
        Compute G + l2 * I - this is equivalent to adding l2-regularisation when G is the Gramian in an OLS problem.
    '''
    return G + l2 * np.identity(G.shape[0])

def EASEr(X, l2 = 500.0):
    ''''
        Compute linear regression solution with Lagrangian multipiers as per (Steck, WWW '19).
        Note: the final model needs to be computed with B /= -np.diag(B) (Lagrange Multiplier)
        Dyn-EASEr updates work on B directly, so this step should be done afterwards.
    '''
    G = compute_gramian(X)
    G_with_l2 = add_diagonal(G, l2) 
    B = inv(G_with_l2)
    return G, B

def compute_diff(X_curr, G_curr, df, new_ts):
    '''
        Compute differences for user-item matrix X and Gramian matrix G.
    '''
    # Filter out all data up to new timestamp; generate X and G
    new_df = df.loc[df.timestamp <= new_ts]
    X_new = generate_csr(new_df, X_curr.shape)
    G_new = compute_gramian(X_new)
    
    # Compute and return differences
    G_diff = G_new - G_curr
    X_diff = X_new - X_curr

    return X_diff, G_diff

def dyngram(new_df, X):
    '''
        Incorporate pageviews in `new_df` into the user-item matrix `X` and return a sparse G_{\Delta}.
    '''
    # Placeholder for row and column indices for G_{\Delta}
    r, c = [], []

    # For every new interaction
    for row in new_df.itertuples():
        # For every item already seen by this user
        for item in X[row.session,:].nonzero()[1]:
            # Update co-occurrence at (i,j)
            r.extend([row.item, item])
            c.extend([item, row.item])
        # Update occurrence for item at (i,i)
        r.append(row.item)
        c.append(row.item)
        X[row.session, row.item] = 1.

    return X, csr_matrix((np.ones_like(r, dtype=np.float64), (r,c)), shape = (X.shape[1],X.shape[1])) 

def dynEASEr(S, G_diff, k):
    '''
        Perform Woodbury update on matrix S = G^{-1}, with G_diff.
    '''
    # Compute real eigen-values
    vals, vecs = eigsh(G_diff, k = k)
    # vals, vecs = eigh(G_diff[nnz,:][:,nnz].todense(), k = k)
    vals, vecs = deepcopy(vals.real), deepcopy(vecs.real)
    
    # Update S through the Woodbury Identity
    C = np.eye(k)
    C[np.diag_indices(k)] /= vals
    VAinv = vecs.T @ S
    return S - (S @ vecs) @ inv(C + VAinv@vecs, overwrite_a=True, check_finite=False) @ VAinv

def incremental_updates(df, X_init, G_init, S_init, init_ts, num_days, update_minutes, rank='exact'):
    '''
        Perform incremental updates on `X_init`, `G_init` and `S_init`, every `update_minutes` for `num_days` after `init_ts`.
    '''
    # Initialise loop variables
    X_curr = lil_matrix(X_init)
    curr_ts = init_ts
    S_curr = S_init.copy()

    # Train dyn-EASEr on specified increments
    timestamps = [
        init_ts + timedelta(minutes=i) for i in range(update_minutes, (60 * 24 * num_days)+1, update_minutes)
    ]

    # Initialise result placeholders
    timings = []
    ranks = []

    # Loop over timestamps
    pbar = tqdm(timestamps)
    for new_ts in pbar:
        # Extract P_diff
        new_df = df.loc[df.timestamp > curr_ts]
        new_df = new_df.loc[new_df.timestamp <= new_ts]
        # Summary statistics for P_diff
        n_users = new_df.session.nunique() 
        n_items = new_df.item.nunique()
        n_views = len(new_df)
            
        # Start timer
        start_overall = time.perf_counter()
        
        # DynGram -- very fast
        X_curr, G_diff = dyngram(new_df, X_curr)
        
        # Extract non-zero rows from Gramian diff
        nnz = list(set(G_diff.nonzero()[0]))

        # Only update if at least one non-zero entry in the Gramian diff
        if nnz:
            # Compute rank on minimal version of Gramian diff
            # Or Estimate rank with interpolative method
            if rank == 'exact':
                k = np.linalg.matrix_rank(G_diff[nnz,:][:,nnz].todense(),hermitian=True)
            elif rank == 'estimate':
                k = estimate_rank(G_diff[nnz,:][:,nnz].todense(), .001)
            elif rank == 'user_bound':
                k = 2 * n_users 
            elif rank == 'item_bound':
                k = 2 * n_items 
            else:
                raise ValueError(f"Invalid choice for rank estimation: {rank}")

            # Update progress-bar
            pbar.set_postfix({'|P|': n_views, '|U|': n_users, '|I|': n_items, 'rank': k})


            # Update S through dynEASEr
            # Avoid function call because it triggers garbage collection for temporary variables, skewing runtime
            # Compute real eigen-values
            vals, vecs = eigsh(G_diff, k = k)
            vals, vecs = deepcopy(vals.real), deepcopy(vecs.real)

            # Update S through the Woodbury Identity
            VAinv = vecs.T @ S_curr
            S_curr -= (S_curr @ vecs) @ inv(np.diag(1.0 / vals) + VAinv@vecs) @ VAinv 

            # Capture and store timings
            end_overall = time.perf_counter()
            time_overall = end_overall - start_overall
            timings.append(time_overall)
            ranks.append(k)
        
        else:
            # Default values when update is empty
            timings.append(0)
            ranks.append(0)

        # Update loop variables
        curr_ts = new_ts
    
    return timestamps, timings, ranks
