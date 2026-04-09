# DocuMind — RAG Pipeline with LangChain & ChromaDB
A modular document ingestion pipeline built with LangChain that loads, splits, embeds, and stores PDF documents into a persistent vector store — foundation for a full Retrieval-Augmented Generation (RAG) system.

## Features
- 📄 PDF loading with PyPDFLoader
- ✂️ Smart text splitting with RecursiveCharacterTextSplitter + tiktoken
- 🧠 Local embeddings with HuggingFace sentence-transformers
- 💾 Persistent vector storage with ChromaDB

## Project Structure
├── indexing/
│   ├── loader.py        # PDF loading logic
│   ├── splitter.py      # Text splitting logic
│   └── embedder.py      # Embedding + vector store logic
├── main.py              # Entry point
├── data/                # Place your PDF file here
└── chroma_db/           # Auto-generated vector store

## Setup
pip install -r requirements.txt
python main.py