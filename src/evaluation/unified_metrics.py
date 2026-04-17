#src/evaluation/unified_metrics.py
import pandas as pd
import numpy as np
import logging
import math

logging.basicConfig(level=logging.INFO)

def evaluate_unified(recommendations_dict, test_df, k=10):
    """
    Unified Evaluation Framework for Leave-Last-Out Recommender Systems.
    """
    logging.info(f"Starting unified evaluation for Top-{k} recommendations...")
    
    hits = 0
    total_users = 0
    
    sum_precision = 0.0
    sum_recall = 0.0
    sum_map = 0.0
    sum_ndcg = 0.0
    
    # Convert test_df into a dictionary for O(1) fast lookups
    # { 'user_string_id': 'item_string_id_they_bought_next' }
    test_dict = dict(zip(test_df['user_id'], test_df['product_id']))
    
    for user_id, rec_list in recommendations_dict.items():
        # Only evaluate users that actually exist in our test set
        if user_id not in test_dict:
            continue
            
        total_users += 1
        target_item = test_dict[user_id]
        
        # Check if the target item is in the top K recommendations
        if target_item in rec_list:
            hits += 1
            # Find the 1-based rank (index + 1)
            rank = rec_list.index(target_item) + 1
            
            # Math formulas for exactly 1 hidden item
            sum_precision += 1.0 / k
            sum_recall += 1.0  
            sum_map += 1.0 / rank
            sum_ndcg += 1.0 / math.log2(rank + 1)
            
    if total_users == 0:
        logging.warning("No overlapping users found between recommendations and test set!")
        return {}
        
    metrics = {
        f"Precision@{k}": sum_precision / total_users,
        f"Recall@{k}": sum_recall / total_users,
        f"MAP@{k}": sum_map / total_users,
        f"NDCG@{k}": sum_ndcg / total_users
    }
    
    return metrics