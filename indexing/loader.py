from langchain_community.document_loaders import PyPDFLoader

def load_document(file_path: str):
    """Load a PDF file and return a list of documents."""
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    print(f"Loaded {len(docs)} pages from {file_path}")
    return docs