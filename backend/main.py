import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from backend import documents
from generation.chain import build_chain, strip_thinking
from indexing.embedder import get_embeddings, load_vectorstore
from indexing.loader import load_document
from indexing.splitter import split_documents

load_dotenv()

CHROMA_DIR = "./chroma_db"
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "http://localhost:5173")

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    embeddings = get_embeddings()
    vectorstore = load_vectorstore(embeddings, CHROMA_DIR)
    state["vectorstore"] = vectorstore
    state["chain"] = build_chain(vectorstore)
    yield
    state.clear()


app = FastAPI(title="DocuMind API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_ORIGIN],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str


@app.get("/api/documents")
def list_documents():
    return documents.list_documents()


@app.post("/api/upload")
async def upload_document(file: UploadFile):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    content = await file.read()
    entry = documents.save_document(file.filename, content)

    file_path = documents.get_document_path(entry["id"])
    docs = load_document(str(file_path))
    split_text = split_documents(docs)
    for chunk in split_text:
        chunk.metadata["document_id"] = entry["id"]
    state["vectorstore"].add_documents(split_text)

    return entry


@app.get("/api/documents/{doc_id}/file")
def get_document_file(doc_id: str):
    path = documents.get_document_path(doc_id)
    if path is None:
        raise HTTPException(status_code=404, detail="Document not found.")
    return FileResponse(path, media_type="application/pdf")


@app.delete("/api/documents/{doc_id}", status_code=204)
def delete_document(doc_id: str):
    entry = documents.delete_document(doc_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="Document not found.")
    state["vectorstore"].delete(where={"document_id": doc_id})


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    raw_answer = state["chain"].invoke(request.question)
    return ChatResponse(answer=strip_thinking(raw_answer))
