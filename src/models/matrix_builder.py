#src/models/matrix_builder.py

from scipy.sparse import csr_matrix
import numpy as np
import logging

def build_matrix(df, n_users, n_items):
    """
    Build sparse matrix using GLOBAL dimensions
    """

    logging.info("Building interaction matrix...")

    matrix = csr_matrix(
        (
            df["rating"],
            (df["user_idx"], df["item_idx"])
        ),
        shape=(n_users, n_items)
    )

    logging.info(f"Matrix shape: {matrix.shape}")

    return matrix

