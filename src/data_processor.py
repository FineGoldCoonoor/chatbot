import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from src.config import PDF_FOLDER_PATH, INDEX_SAVE_PATH, EMBEDDING_MODEL_NAME

def build_and_save_index():
    """
    Loads PDFs, chunks them, creates embeddings, and saves a FAISS index to disk.
    """
    print("🚀 Starting the index building process...")

    pdf_files = [os.path.join(PDF_FOLDER_PATH, f) for f in os.listdir(PDF_FOLDER_PATH) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print(f"❌ No PDF files found in the '{PDF_FOLDER_PATH}' folder. Exiting.")
        return

    print(f"📚 Found {len(pdf_files)} PDF files to process.")
    
    all_pages = []
    for filepath in pdf_files:
        try:
            loader = PyMuPDFLoader(file_path=filepath)
            all_pages.extend(loader.load())
            print(f"  - Successfully loaded {filepath}")
        except Exception as e:
            print(f"⚠️ Error loading {filepath}: {e}")
            continue

    if not all_pages:
        print("❌ No documents could be loaded from the PDF files. Exiting.")
        return

    print(f"📄 Loaded a total of {len(all_pages)} pages.")
    print("Splitting documents into manageable chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunked_docs = text_splitter.split_documents(all_pages)
    print(f"✅ Created {len(chunked_docs)} text chunks.")

    print(f"🧠 Initializing embedding model: {EMBEDDING_MODEL_NAME}...")
    embeddings_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'}, # Use 'cuda' if you have a GPU
        encode_kwargs={'normalize_embeddings': True}
    )

    print("🛠️ Creating FAISS vector store from chunks... (This may take a few minutes)")
    vector_db = FAISS.from_documents(chunked_docs, embeddings_model)
    
    print(f"💾 Saving index to '{INDEX_SAVE_PATH}'...")
    vector_db.save_local(INDEX_SAVE_PATH)
    
    print("\n✅ SUCCESS! The FAISS index has been built and saved.")