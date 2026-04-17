# src/models/time_split.py

import pandas as pd
import logging

def time_based_split(df: pd.DataFrame):
    """
    Perform leave-last-out split per user.
    
    For each user:
    - Sort interactions by timestamp
    - Last interaction → test
    - Remaining → train
    """

    logging.info("Performing time-based split (leave-last-out)...")

    # Sort by user and time
    df = df.sort_values(by=["user_idx", "timestamp"])

    # Get last interaction index per user
    last_idx = df.groupby("user_idx").tail(1).index

    # Split
    test_df = df.loc[last_idx]
    train_df = df.drop(last_idx)

    logging.info(f"Train shape: {train_df.shape}")
    logging.info(f"Test shape: {test_df.shape}")

    return train_df, test_df
