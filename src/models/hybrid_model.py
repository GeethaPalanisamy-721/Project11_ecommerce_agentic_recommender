# src/models/hybrid_model.py
import logging

logging.basicConfig(level=logging.INFO)

def get_popular_items(train_df, idx_to_item, k=10):
    """
    Calculates the top K most popular items across the entire store.
    Used as a cold-start fallback.
    """
    logging.info("Calculating global popularity recommendations...")
    # Count how many times each item was interacted with
    top_item_indices = train_df['item_idx'].value_counts().head(k).index.tolist()
    return [idx_to_item[i] for i in top_item_indices if i in idx_to_item]

def safe_merge_recs(list1, list2, k=10):
    """
    Safely merges two recommendation lists by interleaving them.
    Preserves ranking order and prevents duplicates.
    """
    merged = []
    seen = set()
    
    # Alternate picking from list1 and list2
    for i in range(max(len(list1), len(list2))):
        if i < len(list1) and list1[i] not in seen:
            merged.append(list1[i])
            seen.add(list1[i])
        if len(merged) == k: break
            
        if i < len(list2) and list2[i] not in seen:
            merged.append(list2[i])
            seen.add(list2[i])
        if len(merged) == k: break
            
    return merged

def generate_hybrid_recommendations(user_segments_df, train_df, als_recs, content_recs, idx_to_item, idx_to_user, k=10):
    """
    Routes users to specific recommendation strategies based on their K-Means segment.
    """
    logging.info("Generating Segment-Routed Hybrid Recommendations...")
    
    hybrid_recs = {}
    popularity_recs = get_popular_items(train_df, idx_to_item, k=k)
    
    # FIX: Safely pull the user identifier out of the index
    df_reset = user_segments_df.reset_index()
    # Find the column containing the user index (usually 'user_idx' or 'index')
    user_col = 'user_idx' if 'user_idx' in df_reset.columns else df_reset.columns[0]
    
    # Build a fast lookup dictionary: real_string_user_id -> Segment
    segment_lookup = {}
    for _, row in df_reset.iterrows():
        u_idx = row[user_col]
        segment = row['Segment']
        # Translate the integer back to the Amazon string ID
        real_user_id = idx_to_user.get(u_idx, str(u_idx))
        segment_lookup[real_user_id] = segment
    
    all_target_users = set(als_recs.keys()).union(set(content_recs.keys()))
    
    for user_id in all_target_users:
        segment = segment_lookup.get(user_id, "New/Moderate Users") # Fallback
        
        u_als = als_recs.get(user_id, [])
        u_content = content_recs.get(user_id, [])
        
        # --- THE ROUTING LOGIC ---
        if segment == "High-Value Frequent Raters":
            final_recs = u_als[:k] if u_als else popularity_recs
            
        elif segment == "Occasional Buyers":
            final_recs = safe_merge_recs(u_als, u_content, k=k)
            
        elif segment == "Lapsed Users":
            final_recs = u_content[:k] if u_content else popularity_recs
            
        elif segment == "New/Moderate Users":
            final_recs = popularity_recs[:k]
            
        else:
            final_recs = popularity_recs[:k]
            
        if len(final_recs) < k:
            final_recs = safe_merge_recs(final_recs, popularity_recs, k=k)
            
        hybrid_recs[user_id] = final_recs

    logging.info(f"Hybrid Engine generated recommendations for {len(hybrid_recs)} users.")
    return hybrid_recs