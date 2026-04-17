import streamlit as st
import pandas as pd
import random
import sys
import os
import ast
import re

# Import our backend tools
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
from models.agentic_recommender import recommend_for_user, search_products

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI E-Commerce Recommender", page_icon="🛒", layout="wide")

st.title("🛒 AI-Powered E-Commerce Recommendation Engine")
st.markdown("Built by **Geetha** | Powered by SVD Collaborative Filtering & ChromaDB RAG")
st.markdown("---")

# --- SIDEBAR: PASSIVE RECOMMENDATIONS (WARM USERS) ---
with st.sidebar:
    st.header("👤 User Profile")
    st.write("Simulate a returning user to get collaborative filtering (SVD) recommendations.")
    
    user_id_input = st.text_input("Enter User ID", placeholder="e.g., A123456789")
    
    if st.button("Get Recommendations"):
        if user_id_input:
            with st.spinner("Running SVD Model..."):
                # FIX: We use .func to bypass the CrewAI Tool wrapper and call the python function directly!
                raw_results = recommend_for_user.func(user_id=user_id_input)
                
                try:
                    recs = ast.literal_eval(raw_results)
                    st.success("Recommendations Found!")
                    for item in recs:
                        with st.container():
                            st.markdown(f"**{item['title']}**")
                            st.caption(f"Category: {item['category']} | Price: ${item['price']}")
                            if item['in_stock']:
                                st.write("✅ In Stock")
                            else:
                                st.write("❌ Out of Stock")
                            st.divider()
                except Exception as e:
                    st.error(f"Error parsing recommendations: {e}")
        else:
            st.warning("Please enter a User ID.")

# --- MAIN WINDOW: ACTIVE RAG SEARCH ---
st.header("🔍 Semantic Product Search (RAG)")
st.write("Search for products using natural language. The AI will understand intent and budget!")

query = st.text_input("What are you looking for?", placeholder="e.g., best budget mechanical keyboard for programming under 80 dollars")

if st.button("Search Database"):
    if query:
        with st.spinner("Searching ChromaDB Vector Space..."):
            # Extract price using Regex
            match = re.search(r'(?:under|below)\s+(?:rs\.?|rupees|dollars?|[$₹£€])?\s*(\d+(?:,\d+)*(?:\.\d+)?)', query.lower())
            price_limit = match.group(1).replace(',', '') if match else "99999.0"
            
            # FIX: We use .func to bypass the CrewAI Tool wrapper!
            raw_results = search_products.func(query=query, max_price=price_limit, k=5)
            
            try:
                search_recs = ast.literal_eval(raw_results)
                if not search_recs or isinstance(search_recs, str):
                     st.info("No products found matching those exact criteria.")
                else:
                    st.success(f"Top {len(search_recs)} AI Matches from Catalog:")
                    
                    # Display results in a nice grid
                    cols = st.columns(len(search_recs))
                    for idx, col in enumerate(cols):
                        item = search_recs[idx]
                        with col:
                            st.markdown(f"**Result #{idx+1}**")
                            st.caption(f"Category: {item['category']} | Price: ${item['price']}")
                            st.write(item['description'][:150] + "...")
                            st.button("View Product", key=f"btn_{idx}")
            except Exception as e:
                st.error("Could not find matching products. Try a different query.")
    else:
        st.warning("Please enter a search query.")