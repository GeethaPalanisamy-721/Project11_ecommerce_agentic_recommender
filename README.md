# 🛒 E-Commerce Intelligent Recommendation System

## 🎯 Problem Statement

E-commerce platforms suffer from low conversion due to **generic recommendations** and inability to handle diverse user behaviors.

This system solves for:

* **Warm users** → personalized recommendations from historical data
* **Cold users** → natural language product search

**Core Challenge:**
Amazon Electronics dataset with **~7.8M interactions and 99.97% sparsity**, making recommendation extremely difficult.

---

## 🚀 Solution Overview

A **Hybrid Recommendation System orchestrated by a Multi-Agent AI layer**, combining:

* **Multi-Agent Intent Routing (CrewAI):**
  LLM-powered *Planner, Executor, Critic* agents dynamically route queries and enforce business rules

* **Collaborative Filtering (Implicit ALS):**
  Matrix factorization for personalized recommendations (warm users)

* **Semantic Product Search (RAG):**
  ChromaDB + sentence-transformers for natural language queries (cold users)

* **User Segmentation (K-Means):**
  RFM-based clustering to model user behavior patterns

---

## 🧠 Multi-Agent System Architecture

```
User Input (Streamlit UI)
   ↓
[Planner Agent] → Detects Intent (Recommendation vs Search)
   ↓
[Executor Agent] → Routes to appropriate pipeline
   ├── Path A: ALS Model (Collaborative Filtering)
   └── Path B: Vector DB (Semantic Search via RAG)
   ↓
[Critic Agent] → Applies Business Rules
   • Removes out-of-stock items
   • Enforces price constraints
   • Limits category duplicates
   ↓
Top-K Final Recommendations
```

---

## 📊 Dataset & Scale

* **7.8M interactions**
* **155K users (after K-core filtering)**
* **31K products**
* **Sparsity:** 99.97%

---

## ⚙️ Key Engineering Decisions

* **Temporal Validation (No Leakage):** Leave-last-out split
* **K-Core Filtering:** Users ≥ 5, Items ≥ 10 interactions
* **Sparse Matrix Optimization:** Efficient large-scale handling
* **Metadata Processing:** Robust parsing of noisy Amazon product data
* **Cold-Start Handling:** Content + semantic embeddings

---

## 🤖 Models & Performance

| Model             | Precision@10 | Recall@10  | MAP@10     | NDCG@10    |
| ----------------- | ------------ | ---------- | ---------- | ---------- |
| Matrix SVD        | 0.0018       | 0.0180     | 0.0079     | 0.0102     |
| Surprise SVD      | 0.0001       | 0.0007     | 0.0002     | 0.0003     |
| **Implicit ALS**  | **0.0030**   | **0.0302** | **0.0124** | **0.0165** |
| Content-Based     | 0.0005       | 0.0053     | 0.0022     | 0.0029     |
| **Hybrid System** | 0.0023       | 0.0228     | 0.0092     | 0.0123     |

---

## 🔍 Key Insights

* **Implicit ALS outperforms SVD (~70% higher recall)** on sparse implicit data
* Content-based system enables **cold-start recommendations**
* Hybrid system balances **accuracy, coverage, and flexibility**
* Multi-agent routing enables **real-world decision logic beyond ML models**

---

## 🛠️ Tech Stack

* **ML & Recommenders:** `implicit`, `scikit-surprise`, `scipy`
* **NLP & Embeddings:** `sentence-transformers`, `sklearn`
* **Vector DB:** `ChromaDB`
* **Agent Framework:** `CrewAI` + Local LLM (Llama)
* **Data Processing:** `pandas`, `numpy`
* **UI:** `Streamlit`

---

## ▶️ How to Run

```bash
git clone <repo-url>
cd ecommerce-recommendation-system
pip install -r requirements.txt

# Run the interactive dashboard
streamlit run app.py
```

---

## 📌 Future Improvements

* Bayesian Personalized Ranking (BPR)
* Advanced hybrid re-ranking
* Real-time recommendation API
* LLM upgrade (GPT / Claude for better reasoning)

---

## 💡 Key Takeaway

This project demonstrates how to build a **production-grade, hybrid recommendation system** that:

* Handles extreme sparsity
* Supports both behavioral and semantic queries
* Uses agent-based orchestration for real-world decision making

---

