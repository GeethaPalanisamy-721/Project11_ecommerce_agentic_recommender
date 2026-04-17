# main.py
import logging
import os
import pandas as pd
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from src.data.load_data import load_raw_data, inspect_data, clean_data
from src.data.preprocess import (
    encode_ids,
    iterative_k_core,
    build_sparse_matrix,
    compute_density,
    reindex_ids,
    save_mappings
)
from src.features.rfm_features import compute_rfm, scale_rfm
from src.models.time_split import time_based_split
from src.models.matrix_builder import build_matrix
from src.models.svd_model import train_svd, reconstruct_matrix
from src.models.recommender import recommend_top_k

# Collaborative Filtering Models
from src.models.als_model import train_als_model, generate_als_recommendations
from src.models.surprise_svd import prepare_surprise_data, train_surprise_svd, generate_predictions as surprise_recommend

# Segmentation
from src.models.user_segmentation import evaluate_kmeans, segment_users

# NEW DAY 3 IMPORTS: Content Model & Unified Evaluation
from src.models.content_model import generate_content_recommendations
from src.evaluation.unified_metrics import evaluate_unified
from src.models.hybrid_model import generate_hybrid_recommendations

# Ensure output folders exist
os.makedirs("data/processed", exist_ok=True)
os.makedirs("data/features", exist_ok=True)

logging.basicConfig(level=logging.INFO)

def main():
    logging.info("Starting Recommendation System Pipeline")

    # -----------------------------
    # STEPS 1-6: DATA PREP & SPLIT
    # -----------------------------
    df = load_raw_data("data/raw/ratings_Electronics.csv")
    df = clean_data(df)
    df, user_cats, item_cats = encode_ids(df)
    df_filtered = iterative_k_core(df)
    df_filtered, user_cats, item_cats = reindex_ids(df_filtered)
    save_mappings(user_cats, item_cats)
    df_filtered.to_csv("data/processed/filtered.csv", index=False)

    matrix = build_sparse_matrix(df_filtered)
    density = compute_density(matrix)

    rfm = compute_rfm(df_filtered)
    rfm_scaled = scale_rfm(rfm)
    
    n_users = df_filtered["user_idx"].max() + 1
    n_items = df_filtered["item_idx"].max() + 1
    train_df, test_df = time_based_split(df_filtered)

    # Reusable mappings for recommendation functions
    idx_to_user = dict(enumerate(user_cats))
    idx_to_item = dict(enumerate(item_cats))
    
    train_matrix = build_matrix(train_df, n_users, n_items)

    # -----------------------------
    # STEP 7: SURPRISE SVD PIPELINE
    # -----------------------------
    logging.info("Starting Surprise SVD pipeline...")
    trainset = prepare_surprise_data(train_df)
    surprise_model = train_surprise_svd(trainset)
    surprise_recs = surprise_recommend(surprise_model, train_df, test_df, k=10)
    
    # DAY 3 UPDATE: Use Unified Evaluator
    surprise_results = evaluate_unified(surprise_recs, test_df, k=10)

    # -----------------------------
    # STEP 8: ALS MODEL PIPELINE
    # -----------------------------
    logging.info("Starting ALS pipeline...")
    als_model = train_als_model(train_matrix)
    als_recs = generate_als_recommendations(
        als_model, train_matrix, test_df, idx_to_user, idx_to_item, k=10
    )
    
    # DAY 3 UPDATE: Use Unified Evaluator
    als_results = evaluate_unified(als_recs, test_df, k=10)

    # -----------------------------
    # STEP 9: CONTENT-BASED MODEL (DAY 3)
    # -----------------------------
    logging.info("Starting Content-Based pipeline...")
    
    # Ensure vectors exist before running
    if not os.path.exists("data/features/dense_embeddings.npy"):
        logging.error("Missing dense embeddings! Please run vectorizer.py first.")
        return
        
    content_recs = generate_content_recommendations(
        train_df=train_df,
        embeddings_path="data/features/dense_embeddings.npy",
        idx_to_user=idx_to_user,
        idx_to_item=idx_to_item,
        k=10
    )
    
    # DAY 3 UPDATE: Use Unified Evaluator
    content_results = evaluate_unified(content_recs, test_df, k=10)

    # -----------------------------
    # STEP 10: USER SEGMENTATION
    # -----------------------------
    logging.info("Starting User Segmentation...")
    final_segments = segment_users(rfm_scaled, rfm, k=4)

    # -----------------------------
    # STEP 11: HYBRID SEGMENTED ENGINE
    # -----------------------------
    logging.info("Running Segment-Routed Hybrid Engine...")
    hybrid_recs = generate_hybrid_recommendations(
        user_segments_df=final_segments,
        train_df=train_df,
        als_recs=als_recs,
        content_recs=content_recs,
        idx_to_item=idx_to_item,
        idx_to_user=idx_to_user,
        k=10
    )
    
    hybrid_results = evaluate_unified(hybrid_recs, test_df, k=10)

    # -----------------------------
    # STEP 12: DAY 3 FINAL COMPARISON TABLE
    # -----------------------------
    logging.info("\n" + "="*65)
    logging.info("             DAY 3: FINAL HYBRID MODEL COMPARISON")
    logging.info("="*65)
    
    df_results = pd.DataFrame([
        {"Model": "Surprise SVD", **surprise_results},
        {"Model": "Implicit ALS", **als_results},
        {"Model": "Content-Based (AI)", **content_results},
        {"Model": "Segment-Routed Hybrid", **hybrid_results} # <-- THE GRAND FINALE
    ])
    
    logging.info(f"\n{df_results.to_string(index=False)}")
    logging.info("="*65)
    logging.info("Day 3 Complete!")

    

if __name__ == "__main__":
    main()