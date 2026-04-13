from indexing.loader import load_document
from indexing.splitter import split_documents
from indexing.embedder import get_embeddings, create_vectorstore, load_vectorstore
import os

FILE_PATH = "./data/example.pdf"
CHROMA_DIR = "./chroma_db"

# Step 1 — Load
docs = load_document(FILE_PATH)

# Step 2 — Split
split_text = split_documents(docs, chunk_size=50, chunk_overlap=0)

# Step 3 — Embed + Store
embeddings = get_embeddings()

# create fresh vectorstore or load existing one
if os.path.exists(CHROMA_DIR):
    vectorstore = load_vectorstore(embeddings, CHROMA_DIR)
else:
    vectorstore = create_vectorstore(split_text, embeddings, CHROMA_DIR)
