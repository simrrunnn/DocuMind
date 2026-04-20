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

    prompt = ChatPromptTemplate.from_template("""
    You are a helpful assistant. Answer the question based only on the context below.
    If you don't know the answer, just say "I don't know".
    Context: {context}
    Question: {question}
    Answer:
    """)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

rag_chain = load_chain()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if query := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = rag_chain.invoke(query)
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
'''without streamlit

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
CHROMA_DIR = "./chroma_db"
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")


if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY is not set. Add it to your environment or .env file.")

# Step 1 — LLM (replace with your model)
llm = ChatOpenAI(
    model="openrouter/free",
    openai_api_key=str(OPENROUTER_API_KEY),
    openai_api_base="https://openrouter.ai/api/v1"
)


# Step 2 — Retriever
embeddings = get_embeddings()
vectorstore = load_vectorstore(embeddings, CHROMA_DIR)
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# Step 3 — Prompt
prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. Answer the question based only on the context below.
If you don't know the answer, just say "I don't know".

Context: {context}

Question: {question}

Answer:
""")

# Step 4 — Format retrieved docs
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Step 5 — Build the chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Step 6 — Query
query = input("Enter something: ")
response = rag_chain.invoke(query)
print(response)

'''