import os
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from src.config import (
    INDEX_SAVE_PATH, EMBEDDING_MODEL_NAME, RERANKER_MODEL_NAME, 
    LLM_MODEL_NAME, RETRIEVER_TOP_K, RERANKER_TOP_N
)

def load_vector_db():
    """Loads the FAISS index from disk."""
    if not os.path.exists(INDEX_SAVE_PATH):
        return None
    
    embeddings_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    vector_db = FAISS.load_local(
        INDEX_SAVE_PATH, 
        embeddings_model, 
        allow_dangerous_deserialization=True
    )
    return vector_db

def create_effective_rag_chain(vector_db):
    """Creates the RAG chain with a reranker for improved accuracy."""
    
    # 1. Initialize LLM
    llm = ChatGroq(temperature=0, model_name=LLM_MODEL_NAME)

    # 2. Initialize Reranker
    reranker_model = HuggingFaceCrossEncoder(model_name=RERANKER_MODEL_NAME)
    compressor = CrossEncoderReranker(model=reranker_model, top_n=RERANKER_TOP_N)

    # 3. Create Retriever with Reranking
    base_retriever = vector_db.as_retriever(search_kwargs={"k": RETRIEVER_TOP_K})
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=base_retriever
    )

    # 4. Create the Prompt (Using your most detailed version)
    prompt = ChatPromptTemplate.from_template(
        """
        You are an expert assistant for the Thoothukudi District Police.
        Your primary instruction is to answer the user's question based ONLY on the following context.
        If the information is not in the context, you MUST respond with:
        "The answer is not available in the provided documents."
        Do not use any outside knowledge. Be concise, respectful, and helpful.

        <context>
        {context}
        </context>

        Question: {input}
        Answer:
        """
    )

    # 5. Create the RAG Chain
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(compression_retriever, document_chain)

    return retrieval_chain