# src/models/search_engine.py
import chromadb
import re
import logging
from sentence_transformers import SentenceTransformer, CrossEncoder

logging.basicConfig(level=logging.INFO, format='%(message)s')

def parse_user_query(query):
    """
    Extracts constraints like maximum price from natural language.
    Example: 'headphones under 3000 rupees' -> max_price = 3000.0
    """
    max_price = None
    
    # Regex to look for "under <number>" or "below <number>"
    # Handles formats like "3000", "3,000", "80000"
    match = re.search(r'(?:under|below)\s+(?:rs\.?|rupees|[$₹£€])?\s*(\d+(?:,\d+)*(?:\.\d+)?)', query.lower())
    
    if match:
        # Remove commas and convert to float
        clean_number = match.group(1).replace(',', '')
        max_price = float(clean_number)
        logging.info(f"🔍 Parser detected price constraint: Max Price = {max_price}")
        
    return max_price

class ProductSearchEngine:
    def __init__(self, db_path="data/chroma_db"):
        logging.info("Initializing Search Engine...")
        
        # 1. Connect to ChromaDB
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_collection(name="amazon_products")
        
        # 2. Load the primary Embedder (for initial fast retrieval)
        self.embedder = SentenceTransformer('all-mpnet-base-v2')
        
        # 3. Load the Cross-Encoder (for precise reranking)
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        logging.info("Search Engine Ready!\n" + "="*50)

    def search(self, query, top_k=3):
        logging.info(f"\n👤 USER QUERY: '{query}'")
        
        # Step 1: Parse the Query
        max_price = parse_user_query(query)
        
        # Step 2: Build the ChromaDB "Where" Filter
        where_clause = None
        if max_price:
            # $lte means "Less Than or Equal To"
            where_clause = {"price_float": {"$lte": max_price}}

        # Step 3: Embed the Query
        query_vector = self.embedder.encode(query).tolist()
        
        # Step 4: Retrieve Top 20 Candidates
        if where_clause:
            results = self.collection.query(
                query_embeddings=[query_vector],
                n_results=20,
                where=where_clause
            )
        else:
            results = self.collection.query(
                query_embeddings=[query_vector],
                n_results=20
            )

        if not results['documents'][0]:
            logging.info("❌ No products found matching those constraints.")
            return []

        # Extract the candidates
        candidate_docs = results['documents'][0]
        candidate_ids = results['ids'][0]
        
        # Step 5: Cross-Encoder Reranking
        # Create pairs of (Query, Document) for the AI to score
        rerank_pairs = [[query, doc] for doc in candidate_docs]
        rerank_scores = self.reranker.predict(rerank_pairs)
        
        # Sort candidates based on the new Cross-Encoder scores
        # argsort() sorts ascending, so we take the end and reverse it [::-1]
        ranked_indices = rerank_scores.argsort()[::-1]
        
        final_results = []
        logging.info("🏆 TOP 3 RESULTS:")
        for idx in ranked_indices[:top_k]:
            doc = candidate_docs[idx]
            final_results.append(doc)
            logging.info(f" - {doc[:150]}...") # Print a snippet of the result
            
        return final_results

if __name__ == "__main__":
    # Initialize the engine
    engine = ProductSearchEngine()
    
    # Run the 5 Benchmark Queries from the Day 4 Project Specs
    test_queries = [
        "wireless earbuds for running under 3000 rupees",
        "gaming mouse with RGB lighting",
        "external hard drive 2TB for MacBook",
        "noise cancelling headphones for office work",
        "USB hub with fast charging for laptop"
    ]
    
    for q in test_queries:
        engine.search(q, top_k=3)