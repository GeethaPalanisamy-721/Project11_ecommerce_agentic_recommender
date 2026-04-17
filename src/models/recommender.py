#src/models/ recommender.py
import numpy as np

def recommend_top_k(pred_matrix, train_matrix, user_id, k=10):
    """
    Recommend top K items not already interacted
    """

    user_row = pred_matrix[user_id]

    # Remove already interacted items
    seen_items = train_matrix[user_id].nonzero()[1]
    user_row[seen_items] = -np.inf

    # Top K indices
    top_items = np.argsort(user_row)[-k:][::-1]

    return top_items
