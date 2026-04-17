# src/data/metadata_processor.py

import pandas as pd
import logging
import json
import re
import os

logging.basicConfig(level=logging.INFO)

def parse_amazon_price(price_str):
    """
    Extracts the lower bound float from messy Amazon price strings.
    Example: '$29.99' -> 29.99 | '$20.00 - $50.00' -> 20.0
    """
    if pd.isna(price_str) or not isinstance(price_str, str):
        return 0.0
    
    # Regex to find the first sequence of digits with a decimal
    # Removing commas first (e.g., $1,000.00 -> 1000.00)
    match = re.search(r'(\d+\.\d+)', price_str.replace(',', ''))
    if match:
        return float(match.group(1))
    return 0.0

def process_metadata(json_path, filtered_data_path):
    """
    Reads the metadata JSON, but ONLY keeps products that exist in our filtered interactions.
    Cleans categories, parses prices, and builds the combined text string.
    """
    logging.info("Loading valid product IDs from filtered data...")
    df_filtered = pd.read_csv(filtered_data_path)
    
    # Note: Depending on your Day 1 code, the column might be 'product_id' or 'item_id'.
    # Adjust 'product_id' below if your CSV uses a different name for the raw string ID.
    valid_products = set(df_filtered['product_id'].unique())
    logging.info(f"Looking for metadata for {len(valid_products)} unique products...")

    parsed_data = []
    
    logging.info("Parsing metadata JSON line-by-line...")
    # Amazon JSONs are often "JSON-lines" (one JSON object per line)
    # They also sometimes use single quotes (eval format), so we use a robust reader.
    with open(json_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                # ast.literal_eval is safer for older Amazon datasets, but json.loads is faster 
                # if the format is strict JSON. Let's try standard json first.
                item = json.loads(line.strip())
            except json.JSONDecodeError:
                # Fallback for old UCSD datasets that use python dictionary format
                import ast
                try:
                    item = ast.literal_eval(line.strip())
                except:
                    continue

            item_id = item.get('asin', '')
            
            # Skip items we don't care about!
            if item_id not in valid_products:
                continue
                
            # 1. Extract basic fields
            title = item.get('title', '')
            description = item.get('description', '')
            price_raw = item.get('price', '')
            
            # Handle category list
            categories = item.get('categories', [['']])
            # Flatten if it's a list of lists
            if isinstance(categories, list) and len(categories) > 0 and isinstance(categories[0], list):
                cat_list = categories[0]
            elif isinstance(categories, list):
                cat_list = categories
            else:
                cat_list = ['Unknown']
                
            # Extract category_level_1
            cat_level_1 = cat_list[0] if len(cat_list) > 0 else 'Unknown'
            # Full category string for text
            cat_string = " ".join(cat_list)

            # 2. Parse Price
            price_float = parse_amazon_price(price_raw)

            # 3. Create Combined Content Text
            if not description:
                # Objective: "For products with missing descriptions, use the title repeated twice as a fallback"
                combined_text = f"{title} {title} {cat_string}"
            else:
                # Objective: "concatenate the title, description, and category into a single text string"
                combined_text = f"{title} {description} {cat_string}"

            parsed_data.append({
                'product_id': item_id,
                'title': title,
                'category_level_1': cat_level_1,
                'price': price_float,
                'content_text': combined_text
            })

    # Convert to DataFrame
    meta_df = pd.DataFrame(parsed_data)
    logging.info(f"Successfully processed metadata for {len(meta_df)} products.")
    
    # Save the cleaned metadata
    os.makedirs("data/processed", exist_ok=True)
    out_path = "data/processed/cleaned_metadata.csv"
    meta_df.to_csv(out_path, index=False)
    logging.info(f"Cleaned metadata saved to {out_path}")
    
    return meta_df

if __name__ == "__main__":
    # Quick standalone test
    process_metadata(
        json_path="data/raw/meta_Electronics.json", 
        filtered_data_path="data/processed/filtered.csv"
    )