import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from indexing.embedder import get_embeddings, load_vectorstore
from indexing.splitter import split_documents
from indexing.loader import load_document

CHROMA_DIR = "./chroma_db"
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

@st.cache_resource
def load_chain():
    llm = ChatOpenAI(
        model="openrouter/free",
        openai_api_key=str(OPENROUTER_API_KEY),
        openai_api_base="https://openrouter.ai/api/v1"
    )
    embeddings = get_embeddings()
    vectorstore = load_vectorstore(embeddings, CHROMA_DIR)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    prompt = ChatPromptTemplate.from_template("""\
    You are a helpful assistant. Answer the question based only on the context below.
    If you don't know the answer, just say "I don't know".
    Context: {context}
    Question: {question}
    Answer:
    """
    )

    return (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Initialize the RAG chain
rag_chain = load_chain()

# File uploader in the sidebar
st.sidebar.title("Upload a PDF")
uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file:
    # Save the uploaded file temporarily
    temp_file_path = f"./temp_uploaded_file.pdf"
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Load the uploaded PDF
    docs = load_document(temp_file_path)

    # Split the documents
    split_text = split_documents(docs)

    # Load the existing vector store
    embeddings = get_embeddings()
    vectorstore = load_vectorstore(embeddings, CHROMA_DIR)

        # Add the new documents to the existing vector store
    vectorstore.add_documents(split_text)
    st.sidebar.success("PDF uploaded and processed successfully!")

    # Reload the RAG chain with the updated vector store
    load_chain.clear()
    rag_chain = load_chain()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if query := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = rag_chain.invoke(query)
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})

