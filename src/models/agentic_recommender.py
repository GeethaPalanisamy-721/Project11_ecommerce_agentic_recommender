# src/models/agentic_recommender.py
import os
import random
import logging
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool

logging.basicConfig(level=logging.INFO, format='%(message)s')

# ==========================================
# 1. CONNECT TO LOCAL LLM
# ==========================================
logging.info("Connecting to local Ollama model...")
local_llm = LLM(
    model="ollama/llama3.2",
    base_url="http://localhost:11434"
)

# ==========================================
# 2. DEFINE THE TOOLS
# ==========================================
@tool("recommend_for_user")
def recommend_for_user(user_id: str, k: int = 10) -> str:
    """
    Fetches personalized product recommendations for a known user_id using the SVD model.
    Use this ONLY when a user_id is provided.
    """
    logging.info(f"⚙️ TOOL TRIGGERED: Running SVD model for User {user_id}")
    
    mock_svd_results = [
        {"id": "B0001", "title": "Sony Wireless Headphones", "category": "Audio", "price": 45.0, "in_stock": random.choice([True, False])},
        {"id": "B0002", "title": "Razer Gaming Mouse", "category": "Accessories", "price": 35.0, "in_stock": random.choice([True, False])},
        {"id": "B0003", "title": "Logitech Mechanical Keyboard", "category": "Accessories", "price": 120.0, "in_stock": True},
        {"id": "B0004", "title": "Dell 27-inch Monitor", "category": "Displays", "price": 250.0, "in_stock": True},
    ]
    return str(mock_svd_results)

@tool("search_products")
def search_products(query: str = "best products", max_price: str = "99999.0", k: int = 10, user_id: str = "") -> str:
    """
    Searches the product database using natural language (RAG). 
    Use this when the user provides a search query.
    """
    logging.info(f"⚙️ TOOL TRIGGERED: Running RAG ChromaDB Search for '{query}'")
    try:
        clean_price = ''.join(c for c in str(max_price) if c.isdigit() or c == '.')
        price_limit = float(clean_price) if clean_price else 99999.0
    except Exception:
        price_limit = 99999.0

    try:
        import chromadb
        from sentence_transformers import SentenceTransformer
        
        client = chromadb.PersistentClient(path="data/chroma_db")
        collection = client.get_collection(name="amazon_products")
        embedder = SentenceTransformer('all-mpnet-base-v2')
        
        query_vector = embedder.encode(query).tolist()
        where_clause = {"price_float": {"$lte": price_limit}} if price_limit < 99999.0 else None
        
        results = collection.query(
            query_embeddings=[query_vector],
            n_results=int(k),
            where=where_clause
        )
        
        if not results['documents'][0]:
            return "No products found matching those constraints."
            
        formatted_results = []
        for i, doc in enumerate(results['documents'][0]):
            meta = results['metadatas'][0][i]
            formatted_results.append({
                "description": doc[:100] + "...",
                "category": meta.get('category_level_1', 'Unknown'),
                "price": meta.get('price_float', 0.0),
                "in_stock": random.choice([True, True, False])
            })
        return str(formatted_results)
    except Exception as e:
        return f"Error executing search: {str(e)}"

# ==========================================
# 3. DEFINE THE AGENTS (CLIENT COMPLIANT)
# ==========================================
planner = Agent(
    role="Recommendation Strategy Planner",
    goal="Use your reasoning to detect intent: 'personalised recommendation' vs 'product search'. Output a strict, direct instruction for the Executor.",
    backstory="You are an AI intent-detector. If the user provides a user_id, they want a personalised recommendation. If they provide a query, they want a product search. If both, they want both. You MUST output your decision as a direct command (e.g., 'Use recommend_for_user with user_id X'). Do not write JSON or code.",
    llm=local_llm,
    verbose=True,
    allow_delegation=False
)

executor = Agent(
    role="Recommendation Executor",
    goal="Read the Planner's command and physically execute the chosen tool(s).",
    backstory="You are an obedient data fetcher. You read the intent command from the Planner and IMMEDIATELY trigger the assigned tool. You do not explain yourself. You just run the tool and return its raw output.",
    tools=[recommend_for_user, search_products],
    llm=local_llm,
    verbose=True,
    allow_delegation=False,
    max_iter=3 
)

critic = Agent(
    role="Recommendation Critic",
    goal="Filter the Executor's raw output. Apply constraints (price, stock, max 3 per category).",
    backstory="You are a strict QA bot. You NEVER invent products. You read the raw list from the Executor, delete out-of-stock items, and format the real items into a clean numbered list.",
    llm=local_llm,
    verbose=True,
    allow_delegation=False
)

# ==========================================
# 4. EXECUTION PIPELINE
# ==========================================
def run_scenario(scenario_name, user_input_description):
    logging.info(f"\n{'='*60}\nRUNNING SCENARIO: {scenario_name}\nINPUT: {user_input_description}\n{'='*60}")
    
    plan_task = Task(
        description=f"Analyze this user request: '{user_input_description}'. Use your LLM reasoning to determine if it needs collaborative filtering (user_id), RAG search (query), or both. Write out a single explicit sentence commanding the Executor on which tool to use and with what arguments.",
        expected_output="A single sentence command like 'Use the recommend_for_user tool with user_id A123456789'.",
        agent=planner
    )
    
    execute_task = Task(
        description="Read the Planner's command from the context. You MUST physically execute the required tool using your tool-calling ability to fetch the product data.",
        expected_output="The raw data list returned by the tool.",
        context=[plan_task], # Context flows correctly from Planner -> Executor
        agent=executor
    )
    
    criticize_task = Task(
        description="""Read the exact products from the Executor. 
        1. Remove any product where 'in_stock' is False.
        2. Keep a maximum of 3 products per 'category'.
        3. Format the remaining REAL products into a clean, numbered list showing the Title/Description and Price.
        If no products exist, output 'No products found.' DO NOT invent products.""",
        expected_output="A clean, numbered list of ONLY the real products provided by the Executor.",
        context=[execute_task], # Context flows correctly from Executor -> Critic
        agent=critic
    )
    
    recs_crew = Crew(
        agents=[planner, executor, critic],
        tasks=[plan_task, execute_task, criticize_task],
        process=Process.sequential,
        verbose=True
    )
    
    result = recs_crew.kickoff()
    logging.info(f"\nFINAL OUTPUT FOR {scenario_name}:\n{result}\n")

if __name__ == "__main__":
    #run_scenario("Scenario 1 (Warm User)", "user_id: A123456789")
    #run_scenario("Scenario 2 (Cold Query)", "query: best budget mechanical keyboard for programming under 80 dollars")
    run_scenario("Scenario 3 (Combined)", "user_id: A123456789 AND query: gaming products")