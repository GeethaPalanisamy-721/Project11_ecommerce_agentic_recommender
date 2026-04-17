#src/models/content_model.py
import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)

def generate_content_recommendations(train_df, embeddings_path, idx_to_user, idx_to_item, k=10, max_users=None):
    logging.info("Loading dense embeddings...")
    embeddings = np.load(embeddings_path)
    
    # ---------------------------------------------------------
    # OPTIMIZATION: Normalize vectors for blazing fast Cosine Similarity
    # ---------------------------------------------------------
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Prevent division by zero for empty rows
    normalized_embeddings = embeddings / norms
    
    logging.info("Building User Profiles and generating recommendations...")
    recommendations = {}
    
    # Get the list of unique users we need to process
    unique_users = train_df['user_idx'].unique()
    limit = len(unique_users) if max_users is None else min(max_users, len(unique_users))
    users_to_process = unique_users[:limit]
    
    # Pre-group training data by user for fast lookups
    user_histories = train_df.groupby('user_idx')
    
    for count, user_idx in enumerate(users_to_process):
        user_data = user_histories.get_group(user_idx)
        
        # All items the user has seen (to filter out later)
        seen_items = set(user_data['item_idx'].tolist())
        
        # ---------------------------------------------------------
        # APPLY THE 4-STAR RULE
        # ---------------------------------------------------------
        liked_items = user_data[user_data['rating'] >= 4.0]['item_idx'].tolist()
        
        # Fallback: if they haven't liked anything >= 4, use all their items
        if len(liked_items) == 0:
            liked_items = list(seen_items)
            
        # ---------------------------------------------------------
        # CREATE AVERAGE PREFERENCE VECTOR
        # ---------------------------------------------------------
        # Fetch the 384-d vectors for the liked items
        liked_vectors = normalized_embeddings[liked_items]
        
        # Average them out to create the user profile
        user_profile = np.mean(liked_vectors, axis=0)
        
        # Re-normalize the user profile
        profile_norm = np.linalg.norm(user_profile)
        if profile_norm > 0:
            user_profile = user_profile / profile_norm
            
        # ---------------------------------------------------------
        # FIND TOP K SIMILAR ITEMS (Cosine Similarity = Dot Product)
        # ---------------------------------------------------------
        # Matrix multiplication: (31785, 384) dot (384,) -> (31785,) scores
        scores = np.dot(normalized_embeddings, user_profile)
        
        # Set scores of already seen items to -1 so they aren't recommended
        scores[list(seen_items)] = -1.0
        
        # Get the indices of the top K scores (argsort sorts ascending, so we take the end and reverse)
        top_k_indices = np.argsort(scores)[-k:][::-1]
        
        # Map integer indices back to string IDs
        user_id = idx_to_user[user_idx]
        recommendations[user_id] = [idx_to_item[i] for i in top_k_indices if i in idx_to_item]
        
        if count > 0 and count % 5000 == 0:
            logging.info(f"Processed {count}/{limit} users for Content-Based")

    logging.info(f"Content-Based recommendation generation complete for {len(recommendations)} users")
    return recommendations