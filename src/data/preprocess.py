# src/data/preprocess.py

import pandas as pd
import logging
import numpy as np
from scipy.sparse import csr_matrix
import pickle
import os


def iterative_k_core(
    df: pd.DataFrame,
    min_user_interactions=5,
    min_item_interactions=10
) -> pd.DataFrame:
    """
    Apply k-core filtering iteratively until dataset stabilizes.
    Ensures every user has at least 'min_user_interactions' ratings
    and every item has at least 'min_item_interactions' ratings.
    """
    logging.info("Starting iterative k-core filtering...")

    prev_shape = None
    iteration = 0

    while True:
        iteration += 1
        logging.info(f"Iteration {iteration}...")

        # Filter users
        user_counts = df['user_idx'].value_counts()
        valid_users = user_counts[user_counts >= min_user_interactions].index
        df = df[df['user_idx'].isin(valid_users)]

        # Filter items
        item_counts = df['item_idx'].value_counts()
        valid_items = item_counts[item_counts >= min_item_interactions].index
        df = df[df['item_idx'].isin(valid_items)]

        current_shape = df.shape
        logging.info(f"Shape after iteration {iteration}: {current_shape}")

        # Stop when no more change
        if prev_shape == current_shape:
            logging.info("k-core converged (no further changes)")
            break

        prev_shape = current_shape

    return df


def build_sparse_matrix(df: pd.DataFrame):
    """
    Build user-item interaction matrix in CSR (Compressed Sparse Row) format.
    Rows = users, Columns = items, Values = ratings.
    """
    logging.info("Building sparse interaction matrix...")

    n_users = df['user_idx'].nunique()
    n_items = df['item_idx'].nunique()

    logging.info(f"Matrix dimensions: Users={n_users}, Items={n_items}")

    matrix = csr_matrix(
        (df['rating'], (df['user_idx'], df['item_idx'])),
        shape=(n_users, n_items)
    )

    logging.info("Sparse matrix created")
    return matrix


def compute_density(matrix):
    """
    Compute sparsity/density of the interaction matrix.
    Density = non-zero entries / total possible entries.
    """
    num_nonzero = matrix.nnz
    total_possible = matrix.shape[0] * matrix.shape[1]

    density = num_nonzero / total_possible

    logging.info(f"Matrix Density: {density:.6f}")
    logging.info(f"Sparsity: {1 - density:.6f}")

    return density


def encode_ids(df: pd.DataFrame):
    """
    Encode raw user_id and product_id into integer indices.
    Returns the updated DataFrame plus category lists.
    """
    logging.info("Encoding user_id and product_id to integer indices...")

    df['user_idx'] = pd.Categorical(df['user_id']).codes
    df['item_idx'] = pd.Categorical(df['product_id']).codes

    user_categories = df['user_id'].astype('category').cat.categories
    item_categories = df['product_id'].astype('category').cat.categories

    logging.info("Encoding complete")
    return df, user_categories, item_categories


def reindex_ids(df: pd.DataFrame):
    """
    Re-map user_idx and item_idx to continuous ranges after filtering.
    Returns the updated DataFrame plus category lists.
    """
    logging.info("Re-indexing user and item IDs...")

    df['user_idx'] = pd.Categorical(df['user_id']).codes
    df['item_idx'] = pd.Categorical(df['product_id']).codes

    user_categories = df['user_id'].astype('category').cat.categories
    item_categories = df['product_id'].astype('category').cat.categories

    logging.info("Re-indexing complete")
    return df, user_categories, item_categories


def save_mappings(user_categories, item_categories, path="artifacts"):
    """
    Save mapping dictionaries (int -> original ID) to disk using pickle.
    Critical for inference: allows translating predictions back to real IDs.
    """
    logging.info("Saving mapping dictionaries...")

    os.makedirs(path, exist_ok=True)

    user_mapping = dict(enumerate(user_categories))
    item_mapping = dict(enumerate(item_categories))

    with open(f"{path}/user_mapping.pkl", "wb") as f:
        pickle.dump(user_mapping, f)
    with open(f"{path}/item_mapping.pkl", "wb") as f:
        pickle.dump(item_mapping, f)

    logging.info("Mappings saved successfully")
