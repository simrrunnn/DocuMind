import os
import re

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

THINKING_BLOCK_RE = re.compile(r"<think(?:ing)?>.*?</think(?:ing)?>", re.IGNORECASE | re.DOTALL)

PROMPT = ChatPromptTemplate.from_template("""\
You are a helpful assistant. Answer the question based only on the context below.
If you don't know the answer, just say "I don't know".
Context: {context}
Question: {question}
Answer:
""")


def strip_thinking(text: str) -> str:
    """Remove any <think>/<thinking> reasoning blocks a model may emit inline."""
    return THINKING_BLOCK_RE.sub("", text).strip()


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def build_chain(vectorstore):
    """Build the retrieval-augmented generation chain for a given vectorstore."""
    llm = ChatOpenAI(
        model="openrouter/free",
        openai_api_key=str(os.getenv("OPENROUTER_API_KEY")),
        openai_api_base="https://openrouter.ai/api/v1",
    )
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | PROMPT
        | llm
        | StrOutputParser()
    )
