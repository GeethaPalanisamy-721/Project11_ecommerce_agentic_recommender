#src/models/surprise_svd.py
import logging
import pandas as pd
import numpy as np
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split as surprise_split


def prepare_surprise_data(train_df: pd.DataFrame):
    """
    Convert pandas DataFrame to Surprise Dataset format.
    """
    logging.info("Preparing data for Surprise SVD...")

    reader = Reader(rating_scale=(1, 5))

    data = Dataset.load_from_df(
        train_df[['user_id', 'product_id', 'rating']],
        reader
    )

    trainset = data.build_full_trainset()

    logging.info("Surprise dataset prepared")

    return trainset


def train_surprise_svd(trainset):
    """
    Train Surprise SVD (Funk SVD with regularization).
    """
    logging.info("Training Surprise SVD model...")

    model = SVD(
        n_factors=50,
        n_epochs=20,
        lr_all=0.005,
        reg_all=0.02,
        random_state=42
    )

    model.fit(trainset)

    logging.info("Surprise SVD training complete")

    return model

def generate_predictions(model, train_df: pd.DataFrame, test_df: pd.DataFrame, k=10, n_candidates=1000):
    """
    Generate top-K recommendations per user using candidate sampling.
    """

    logging.info("Generating recommendations using Surprise SVD (optimized)...")

    user_seen_items = train_df.groupby('user_id')['product_id'].apply(set).to_dict()
    all_items = train_df['product_id'].unique()

    recommendations = {}

    users = test_df['user_id'].unique()

    for idx, user in enumerate(users):
        seen_items = user_seen_items.get(user, set())

        # 🔥 SAMPLE instead of full scan
        candidates = np.random.choice(all_items, size=min(n_candidates, len(all_items)), replace=False)

        scores = []

        for item in candidates:
            if item in seen_items:
                continue

            pred = model.predict(user, item)
            scores.append((item, pred.est))

        # Top-K items
        top_k_items = sorted(scores, key=lambda x: x[1], reverse=True)[:k]
        recommendations[user] = [item for item, _ in top_k_items]

        # Optional progress log (very useful)
        if idx % 5000 == 0:
            logging.info(f"Processed {idx}/{len(users)} users")

    logging.info("Recommendation generation complete")

    return recommendations
