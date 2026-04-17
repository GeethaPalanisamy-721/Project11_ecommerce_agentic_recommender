# src/data/load_data.py

import pandas as pd
import logging
import os
import io

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

# Logging config
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/data_inspection.log", mode="w")
    ]
)


def load_raw_data(file_path: str) -> pd.DataFrame:
    # Load dataset with correct dtypes and memory optimization
    
    df = pd.read_csv(
        file_path,
        header=None,
        names=["user_id", "product_id", "rating", "timestamp"],
        dtype={
            "user_id": "string",
            "product_id": "string",
            "rating": "float32",
            "timestamp": "int64"
        },
        low_memory=False 
    )

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")

    logging.info("Data loaded successfully")
    logging.info(f"Shape: {df.shape}")

    return df


def inspect_data(df: pd.DataFrame) -> None:
    #Log dataset insights
    
    buffer = io.StringIO()
    df.info(buf=buffer)

    logging.info("First 5 rows:\n%s", df.head().to_string())
    logging.info("Info:\n%s", buffer.getvalue())
    logging.info("Missing values:\n%s", df.isnull().sum().to_string())

    logging.info(
        "Unique counts: Users=%d, Products=%d",
        df['user_id'].nunique(),
        df['product_id'].nunique()
    )


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # Remove nulls and duplicates
    before_shape = df.shape
    df = df.dropna().drop_duplicates()

    logging.info("Data cleaned: %s -> %s", before_shape, df.shape)

    return df


def encode_ids(df: pd.DataFrame):
    # Convert user_id and product_id to integer indices 
    
    logging.info("Encoding user_id and product_id to integer indices...")

    df["user_idx"] = df["user_id"].astype("category").cat.codes
    df["item_idx"] = df["product_id"].astype("category").cat.codes

    # Save mappings (VERY IMPORTANT for inference later)
    user_mapping = dict(enumerate(df["user_id"].astype("category").cat.categories))
    item_mapping = dict(enumerate(df["product_id"].astype("category").cat.categories))

    logging.info("Encoding complete")
    logging.info(f"Total users: {len(user_mapping)}, Total items: {len(item_mapping)}")

    return df, user_mapping, item_mapping
