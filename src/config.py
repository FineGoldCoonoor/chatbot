# src/config.py

# --- File Paths ---
PDF_FOLDER_PATH = "data"
INDEX_SAVE_PATH = "faiss_index"

# --- Model Configuration ---
# Embedding model for vectorizing text. "bge-small" is efficient and effective.
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# Reranker model to improve search results. This one is small and fast.
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# LLM for generating the final answer. Llama3-70b is powerful.
LLM_MODEL_NAME = "llama3-70b-8192"

# --- Retrieval Configuration ---
# How many documents the base retriever should fetch.
RETRIEVER_TOP_K = 20

# How many of the best documents the reranker should return to the LLM.
RERANKER_TOP_N = 5 # Increased to 5 for slightly more context