# src/models/user_segmentation.py

import pandas as pd
import numpy as np
import logging
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

logging.basicConfig(level=logging.INFO)

def evaluate_kmeans(rfm_scaled: pd.DataFrame, max_k=8):
    """
    Evaluates KMeans using the Elbow Method and Silhouette Score.
    Uses sampling for Silhouette to prevent memory crashes on 150k+ users.
    """
    logging.info("Evaluating K-Means with Elbow Method and Silhouette Score...")
    
    # Drop user_id index if it exists so we only cluster on features
    features = rfm_scaled.copy()
    if 'user_idx' in features.columns:
        features = features.drop(columns=['user_idx'])

    inertia = []
    sil_scores = []
    
    # Sample for silhouette score to avoid OOM (Out of Memory) errors
    sample_size = min(20000, len(features))
    sample_features = features.sample(n=sample_size, random_state=42)

    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
        
        # Fit full data for inertia
        kmeans.fit(features)
        inertia.append(kmeans.inertia_)
        
        # Fit sample for silhouette
        sample_labels = kmeans.predict(sample_features)
        sil = silhouette_score(sample_features, sample_labels)
        sil_scores.append(sil)
        
        logging.info(f"k={k} | Inertia: {kmeans.inertia_:,.0f} | Silhouette: {sil:.4f}")

    # Save a quick plot for your report!
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(range(2, max_k + 1), inertia, marker='o')
    plt.title('Elbow Method (Inertia)')
    plt.xlabel('Number of clusters (k)')
    
    plt.subplot(1, 2, 2)
    plt.plot(range(2, max_k + 1), sil_scores, marker='o', color='orange')
    plt.title('Silhouette Score')
    plt.xlabel('Number of clusters (k)')
    
    os.makedirs("artifacts", exist_ok=True)
    plt.savefig("artifacts/kmeans_evaluation.png")
    logging.info("Saved K-Means evaluation plot to artifacts/kmeans_evaluation.png")


def segment_users(rfm_scaled: pd.DataFrame, rfm_raw: pd.DataFrame, k=4):
    """
    Applies K-Means (k=4) and maps clusters to business profiles.
    """
    logging.info(f"Running Final K-Means with k={k}...")
    
    features = rfm_scaled.copy()
    if 'user_idx' in features.columns:
        features = features.drop(columns=['user_idx'])

    # 1. Fit the Model
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    cluster_labels = kmeans.fit_predict(features)
    
    # 2. Attach labels back to the raw RFM dataframe
    rfm_result = rfm_raw.copy()
    rfm_result['Cluster'] = cluster_labels
    
    # 3. Analyze Centroids
    centroids = rfm_result.groupby('Cluster').mean()
    logging.info(f"Cluster Centroids (Raw values):\n{centroids}")
    
    # 4. Map Clusters to Business Names (Using your explicit dictionary!)
    segment_labels = {
        0: "Occasional Buyers",
        1: "Lapsed Users",
        2: "High-Value Frequent Raters",
        3: "New/Moderate Users" 
    }
            
    rfm_result['Segment'] = rfm_result['Cluster'].map(segment_labels)
    
    # Log the final counts
    logging.info("Final Segment Counts:")
    logging.info(f"\n{rfm_result['Segment'].value_counts()}")
    
    # Save the output
    os.makedirs("data/processed", exist_ok=True)
    rfm_result.to_csv("data/processed/user_segments.csv")
    logging.info("Saved user segments to data/processed/user_segments.csv")
    
    return rfm_result