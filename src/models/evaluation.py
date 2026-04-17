# src/models/evaluation.py
import numpy as np
import logging


def precision_at_k(recommended, ground_truth, k):
    return int(ground_truth in recommended[:k]) / k


def recall_at_k(recommended, ground_truth, k):
    return int(ground_truth in recommended[:k])


def ndcg_at_k(recommended, ground_truth, k):
    if ground_truth in recommended[:k]:
        rank = np.where(recommended[:k] == ground_truth)[0][0] + 1
        return 1 / np.log2(rank + 1)
    return 0


def average_precision_at_k(recommended, ground_truth, k):
    if ground_truth in recommended[:k]:
        rank = np.where(recommended[:k] == ground_truth)[0][0] + 1
        return 1 / rank
    return 0

def evaluate_model(pred_matrix, train_matrix, test_df, k=10, num_users=1000):
    """
    Evaluate model using sampled users
    """

    logging.info("Starting model evaluation...")

    precisions, recalls, ndcgs, maps = [], [], [], []

    users = test_df["user_idx"].unique()[:num_users]

    for user in users:
        # Ground truth item
        true_item = test_df[test_df["user_idx"] == user]["item_idx"].values[0]

        # Get predictions
        user_scores = pred_matrix[user].copy()

        # Remove seen items
        seen_items = train_matrix[user].nonzero()[1]
        user_scores[seen_items] = -np.inf

        # Top-K
        top_k = np.argsort(user_scores)[-k:][::-1]

        # Metrics
        precisions.append(precision_at_k(top_k, true_item, k))
        recalls.append(recall_at_k(top_k, true_item, k))
        ndcgs.append(ndcg_at_k(top_k, true_item, k))
        maps.append(average_precision_at_k(top_k, true_item, k))

    results = {
        "Precision@K": np.mean(precisions),
        "Recall@K": np.mean(recalls),
        "NDCG@K": np.mean(ndcgs),
        "MAP@K": np.mean(maps)
    }

    logging.info(f"Evaluation Results: {results}")

    return results
