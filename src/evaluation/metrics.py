# src/evaluation/metrics.py

import numpy as np
import pandas as pd
import logging


def precision_at_k(recommended, relevant, k):
    return len(set(recommended[:k]) & relevant) / k


def recall_at_k(recommended, relevant, k):
    if len(relevant) == 0:
        return 0
    return len(set(recommended[:k]) & relevant) / len(relevant)


def dcg_at_k(recommended, relevant, k):
    dcg = 0.0
    for i, item in enumerate(recommended[:k]):
        if item in relevant:
            dcg += 1 / np.log2(i + 2)
    return dcg


def ndcg_at_k(recommended, relevant, k):
    ideal_dcg = dcg_at_k(list(relevant), relevant, k)
    if ideal_dcg == 0:
        return 0
    return dcg_at_k(recommended, relevant, k) / ideal_dcg


def average_precision_at_k(recommended, relevant, k):
    score = 0.0
    hits = 0

    for i, item in enumerate(recommended[:k]):
        if item in relevant:
            hits += 1
            score += hits / (i + 1)

    if hits == 0:
        return 0

    return score / min(len(relevant), k)


def evaluate_recommendations(recommendations: dict, test_df: pd.DataFrame, k=10):
    """
    Evaluate recommendation dictionary against test set.

    Args:
        recommendations: {user_id: [item1, item2, ...]}
        test_df: dataframe with ground truth interactions
        k: top-K

    Returns:
        dict of evaluation metrics
    """

    logging.info("Evaluating recommendation model...")

    # Ground truth: leave-last-out → one item per user
    ground_truth = test_df.groupby("user_id")["product_id"].apply(set).to_dict()

    precisions = []
    recalls = []
    ndcgs = []
    maps = []

    for user, recs in recommendations.items():
        if user not in ground_truth:
            continue

        relevant = ground_truth[user]

        precisions.append(precision_at_k(recs, relevant, k))
        recalls.append(recall_at_k(recs, relevant, k))
        ndcgs.append(ndcg_at_k(recs, relevant, k))
        maps.append(average_precision_at_k(recs, relevant, k))

    results = {
        "Precision@K": np.mean(precisions),
        "Recall@K": np.mean(recalls),
        "NDCG@K": np.mean(ndcgs),
        "MAP@K": np.mean(maps)
    }

    logging.info(f"Evaluation Results: {results}")

    return results
