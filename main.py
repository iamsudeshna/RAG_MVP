import os
from typing import Optional
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from modules.doc_loader import load_and_split
from modules.vectordb import save_vectorstore, load_vectorstore
from modules.llm_wrapper import get_llm
from modules.rag_chain import make_rag_chain

app = FastAPI(title="Basic RAG (Ollama + FAISS + Llama2)")

class IngestRequest(BaseModel):
    path: str
    chunk_size: int = 400
    chunk_overlap: int = 50
    embed_model: str = "nomic-embed-text"  # Ollama embeddings model

class AskRequest(BaseModel):
    question: str
    embed_model: str = "nomic-embed-text"
    llm_model: str = "llama2"
    ollama_base_url: Optional[str] = None


@app.get("/")
def root():
    return {"ok": True}


@app.post("/ingest")
def ingest(req: IngestRequest):
    # 1) Load + split docs
    splits = load_and_split(
        path=req.path,
        chunk_size=req.chunk_size,
        chunk_overlap=req.chunk_overlap,
    )
    if not splits:
        raise HTTPException(status_code=400, detail="No text found after splitting.")

    # 2) Build & save FAISS vectorstore
    save_vectorstore(documents=splits, model_name=req.embed_model)

    return {
        "status": "ingested",
        "chunks": len(splits),
        "vectorstore_path": os.environ.get("VECTOR_DB_DIR", "vectorstore/faiss_index")
    }


@app.post("/ask")
def ask(req: AskRequest):
    # 1) Loading vectorstore (uses embed_model internally if not provided)
    try:
        from langchain_community.embeddings import OllamaEmbeddings
        embedder = OllamaEmbeddings(model=req.embed_model)
        vs = load_vectorstore(embedder=embedder, backend="faiss")
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    # 2) Retriever + LLM
    retriever = vs.as_retriever(search_kwargs={"k": 4})
    llm = get_llm(model_name=req.llm_model, base_url=req.ollama_base_url)

    # 3) RAG chain
    chain = make_rag_chain(retriever=retriever, llm=llm)

    # 4) Ask question
    result = chain.invoke({"query": req.question})

    answer = result.get("result")
    sources = [
        {"source": d.metadata.get("source"), "snippet": d.page_content[:160]}
        for d in result.get("source_documents", [])
    ]

    return {"answer": answer, "sources": sources}
