# src/models/rag_indexer.py
import pandas as pd
import numpy as np
import chromadb
import logging
import os
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)

def build_vector_database(metadata_path, ratings_path, db_path="data/chroma_db"):
    logging.info("Starting ChromaDB Indexing Process...")
    
    # 1. Load Metadata and Ratings
    logging.info("Loading metadata and ratings...")
    meta_df = pd.read_csv(metadata_path)
    ratings_df = pd.read_csv(ratings_path)
    
    # 2. Calculate average_rating and total_ratings per product
    logging.info("Calculating rating statistics...")
    stats = ratings_df.groupby('product_id')['rating'].agg(
        average_rating='mean',
        total_ratings='count'
    ).reset_index()
    
    # Merge stats into metadata
    df = pd.merge(meta_df, stats, on='product_id', how='left')
    
    # Fill missing values just in case
    df['average_rating'] = df['average_rating'].fillna(0.0).round(2)
    df['total_ratings'] = df['total_ratings'].fillna(0).astype(int)
    df['price'] = df['price'].fillna(0.0)
    df['category_level_1'] = df['category_level_1'].fillna("Unknown")
    df['content_text'] = df['content_text'].fillna("")
    
    # 3. Create the Structured Text Chunk (Max 500 chars as per spec)
    logging.info("Formatting text chunks...")
    documents = []
    metadatas = []
    ids = []
    
    for _, row in df.iterrows():
        # Build the exact string requested in your Day 4 objectives
        chunk = f"Product: {row['title']}. Category: {row['category_level_1']}. Price: {row['price']}. Description: {row['content_text']}"
        # Truncate to 500 characters
        chunk = chunk[:500]
        
        documents.append(chunk)
        ids.append(str(row['product_id']))
        
        # Metadata dictionary for ChromaDB where-clause filtering
        metadatas.append({
            "category_level_1": row['category_level_1'],
            "price_float": float(row['price']),
            "average_rating": float(row['average_rating']),
            "total_ratings": int(row['total_ratings'])
        })

    # 4. Initialize ChromaDB
    logging.info(f"Connecting to ChromaDB at {db_path}...")
    os.makedirs(db_path, exist_ok=True)
    client = chromadb.PersistentClient(path=db_path)
    
    # Create or get the collection
    collection = client.get_or_create_collection(name="amazon_products")
    
    # 5. Load the stronger Embedding Model
    logging.info("Loading 'all-mpnet-base-v2' model (This may take a moment to download)...")
    model = SentenceTransformer('all-mpnet-base-v2')
    
    logging.info("Encoding documents and uploading to ChromaDB in batches...")
    
    # Batch processing to save memory
    batch_size = 1000
    total_docs = len(documents)
    
    for i in range(0, total_docs, batch_size):
        end_idx = min(i + batch_size, total_docs)
        
        batch_docs = documents[i:end_idx]
        batch_ids = ids[i:end_idx]
        batch_meta = metadatas[i:end_idx]
        
        # Generate embeddings
        batch_embeddings = model.encode(batch_docs, show_progress_bar=False)
        
        # Upsert into ChromaDB
        collection.upsert(
            ids=batch_ids,
            documents=batch_docs,
            metadatas=batch_meta,
            embeddings=batch_embeddings.tolist()
        )
        
        if (i > 0 and i % 5000 == 0) or end_idx == total_docs:
            logging.info(f"Indexed {end_idx} / {total_docs} products...")

    logging.info("✅ ChromaDB Indexing Complete! Your semantic database is ready.")

if __name__ == "__main__":
    build_vector_database(
        metadata_path="data/processed/cleaned_metadata.csv",
        ratings_path="data/processed/filtered.csv"
    )