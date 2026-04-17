# src/features/rfm_features.py

import pandas as pd
import logging
from sklearn.preprocessing import StandardScaler


def compute_rfm(df: pd.DataFrame) -> pd.DataFrame:
    # Compute Recency, Frequency, Monetary features for each user
    logging.info("Computing RFM features...")

    # ----------------------------------
    # Reference date (latest timestamp)
    # ----------------------------------
    reference_date = df["timestamp"].max()

    # ----------------------------------
    # Group by user
    # ----------------------------------
    rfm = df.groupby("user_idx").agg({
        "timestamp": "max",   # last interaction
        "rating": ["count", "mean"]
    })

    # Flatten column names
    rfm.columns = ["last_interaction", "frequency", "monetary"]

    # ----------------------------------
    # Recency calculation
    # ----------------------------------
    rfm["recency"] = (reference_date - rfm["last_interaction"]).dt.days

    # Reorder columns
    rfm = rfm[["recency", "frequency", "monetary"]]

    logging.info(f"RFM computed. Shape: {rfm.shape}")

    return rfm


def scale_rfm(rfm: pd.DataFrame) -> pd.DataFrame:
   #  Standardize RFM features
    logging.info("Scaling RFM features...")

    scaler = StandardScaler()

    rfm_scaled_values = scaler.fit_transform(rfm)

    rfm_scaled = pd.DataFrame(
        rfm_scaled_values,
        index=rfm.index,
        columns=rfm.columns
    )

    logging.info("RFM scaling complete")

    return rfm_scaled
