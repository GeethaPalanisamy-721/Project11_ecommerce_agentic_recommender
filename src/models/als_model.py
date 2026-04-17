#src/models/als_model.py

import logging
import numpy as np
from implicit.als import AlternatingLeastSquares

logging.basicConfig(level=logging.INFO)


def train_als_model(train_matrix, factors=50, regularization=0.1, iterations=20):
    """
    Train ALS model (implicit feedback)
    """
    logging.info("Training ALS model...")

    # CRITICAL FIX: implicit versions >= 0.5.0 expect a USER-ITEM matrix.
    # Do NOT transpose train_matrix. 
    user_item_matrix = train_matrix.tocsr().astype(np.float32)

    model = AlternatingLeastSquares(
        factors=factors,
        regularization=regularization,
        iterations=iterations,
        use_gpu=False  # change to True if GPU available
    )

    model.fit(user_item_matrix)

    logging.info("ALS training complete")
    return model


def generate_als_recommendations(model, train_matrix, test_df, idx_to_user, idx_to_item, k=10, max_users=None):
    logging.info("Generating ALS recommendations...")

    recommendations = {}
    n_users = train_matrix.shape[0]

    limit = n_users if max_users is None else min(max_users, n_users)
    user_item_matrix = train_matrix.tocsr()

    for user_idx in range(limit):
        
        # CRITICAL FIX: Extract ONLY this user's single row
        user_row = user_item_matrix[user_idx]

        ids, scores = model.recommend(
            userid=user_idx,
            user_items=user_row,  # Pass only the 1-row sparse matrix here!
            N=k,
            filter_already_liked_items=True
        )

        user_id = idx_to_user[user_idx]
        recommendations[user_id] = [
            idx_to_item[i] for i in ids.tolist()
            if i in idx_to_item
        ]

        if user_idx > 0 and user_idx % 5000 == 0:
            logging.info(f"Processed {user_idx}/{limit} users")

    logging.info(f"ALS recommendation generation complete for {len(recommendations)} users")

    return recommendations