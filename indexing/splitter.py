from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_documents(docs, encoding_name: str = "o200k_base", chunk_size: int = 50, chunk_overlap: int = 0):
    """Split documents into smaller chunks using tiktoken encoder."""
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name=encoding_name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    split_text = text_splitter.split_documents(docs)
    print(f"Split into {len(split_text)} sub-documents.")
    return split_text