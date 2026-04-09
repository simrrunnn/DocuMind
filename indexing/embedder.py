from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

def get_embeddings(model_name: str = "sentence-transformers/all-mpnet-base-v2"):
    """Initialize and return the HuggingFace embeddings model."""
    return HuggingFaceEmbeddings(model_name=model_name)

def create_vectorstore(split_text, embeddings, persist_directory: str = "./chroma_db"):
    """Create and persist a Chroma vector store from documents."""
    vectorstore = Chroma.from_documents(
        documents=split_text,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    print(f"Stored {vectorstore._collection.count()} chunks in vector store.")
    return vectorstore

def load_vectorstore(embeddings, persist_directory: str = "./chroma_db"):
    """Load an existing Chroma vector store from disk."""
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )
    print(f"Loaded {vectorstore._collection.count()} chunks from vector store.")
    return vectorstore