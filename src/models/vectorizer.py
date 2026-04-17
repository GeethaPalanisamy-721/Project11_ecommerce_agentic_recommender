# src/models/vectorizer.py
import pandas as pd
import numpy as np
import logging
import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from scipy.sparse import save_npz

logging.basicConfig(level=logging.INFO)

def build_content_vectors(metadata_path, mapping_path="artifacts", output_dir="data/features"):
    os.makedirs(output_dir, exist_ok=True)
    
    logging.info("Loading cleaned metadata...")
    df = pd.read_csv(metadata_path)
    df['content_text'] = df['content_text'].fillna("")
    
    # ---------------------------------------------------------
    # FIX 1 & 2: ALIGN WITH ALS ITEM_IDX & HANDLE MISSING METADATA
    # ---------------------------------------------------------
    logging.info("Aligning metadata with global item indices...")
    
    # Load the mapping from Day 1 (item_idx -> product_id)
    with open(f"{mapping_path}/item_mapping.pkl", "rb") as f:
        idx_to_item = pickle.load(f)
        
    n_total_items = len(idx_to_item)
    # Reverse mapping to product_id -> item_idx
    item_to_idx = {v: k for k, v in idx_to_item.items()}
    
    # Map the metadata product_ids to our ALS item_idxs
    df['item_idx'] = df['product_id'].map(item_to_idx)
    
    # Drop any metadata for products that didn't survive Day 1 k-core filtering
    df = df.dropna(subset=['item_idx']).copy()
    df['item_idx'] = df['item_idx'].astype(int)
    
    logging.info(f"Metadata coverage: {len(df)} / {n_total_items} expected items")
    
    # Create a full text corpus aligned exactly to item_idx
    # Missing items will just have an empty string ""
    full_text_corpus = [""] * n_total_items
    for _, row in df.iterrows():
        idx = row['item_idx']
        full_text_corpus[idx] = row['content_text']

    # ---------------------------------------------------------
    # FIX 4: TF-IDF MEMORY OPTIMIZATION & VECTORIZATION
    # ---------------------------------------------------------
    logging.info("Starting TF-IDF Vectorization (Sparse)...")
    tfidf = TfidfVectorizer(
        max_features=5000, 
        ngram_range=(1, 2), 
        stop_words='english',
        dtype=np.float32  # Memory optimization applied!
    )
    # Because full_text_corpus is ordered by item_idx, row 0 = item_idx 0
    tfidf_matrix = tfidf.fit_transform(full_text_corpus)
    
    save_npz(f"{output_dir}/tfidf_matrix.npz", tfidf_matrix)
    logging.info(f"TF-IDF matrix shape: {tfidf_matrix.shape} saved.")
    
    # ---------------------------------------------------------
    # FIX 5: SAVE VECTORIZER FOR DAY 4 (RAG)
    # ---------------------------------------------------------
    with open(f"{output_dir}/tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(tfidf, f)
    logging.info("TF-IDF Vectorizer saved successfully.")

    # ---------------------------------------------------------
    # FIX 3: SENTENCE TRANSFORMERS WITH BATCHING
    # ---------------------------------------------------------
    logging.info("Loading SentenceTransformer model 'all-MiniLM-L6-v2'...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    logging.info("Generating Dense Embeddings...")
    
    # We only encode the texts we actually have to save processing time
    valid_texts = df['content_text'].tolist()
    valid_indices = df['item_idx'].tolist()
    
    # Batching applied (batch_size=64 is usually a sweet spot for CPU/GPU)
    valid_embeddings = model.encode(
        valid_texts, 
        batch_size=64, 
        show_progress_bar=True
    )
    
    # Place the encoded embeddings into the correctly aligned index matrix
    # Shape is (31785, 384) because MiniLM outputs 384-dimensional vectors
    dense_embeddings = np.zeros((n_total_items, 384), dtype=np.float32)
    dense_embeddings[valid_indices] = valid_embeddings
    
    np.save(f"{output_dir}/dense_embeddings.npy", dense_embeddings)
    logging.info(f"Dense embeddings shape: {dense_embeddings.shape} saved.")
        
    logging.info("Vectorization complete! All features safely aligned and saved.")

if __name__ == "__main__":
    build_content_vectors("data/processed/cleaned_metadata.csv")