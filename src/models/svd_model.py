#src/models/svd_model.py
import numpy as np
from scipy.sparse.linalg import svds
import logging

def train_svd(matrix, n_components=50):
    """
    Train Truncated SVD on sparse matrix
    """

    logging.info("Training SVD model...")

    # Compute SVD
    U, sigma, Vt = svds(matrix, k=n_components)

    # Convert sigma to diagonal matrix
    sigma = np.diag(sigma)

    logging.info("SVD training complete")

    return U, sigma, Vt


def reconstruct_matrix(U, sigma, Vt):
    """
    Reconstruct predicted rating matrix
    """

    logging.info("Reconstructing predicted matrix...")

    pred_matrix = np.dot(np.dot(U, sigma), Vt)

    return pred_matrix
