# vectordb.py
import os
from typing import List

from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings

VECTOR_DB_DIR = os.environ.get("VECTOR_DB_DIR", "vectorstore/faiss_index")

def build_vectorstore(documents: List[Document], model_name: str = "nomic-embed-text") -> FAISS:
    embedder = OllamaEmbeddings(model=model_name)
    vectorstore = FAISS.from_documents(documents, embedder)
    print(f"✅ Built FAISS vectorstore with {len(documents)} documents")
    return vectorstore

def save_vectorstore(documents: List[Document], model_name: str = "nomic-embed-text") -> None:
    os.makedirs(VECTOR_DB_DIR, exist_ok=True)
    vectorstore = build_vectorstore(documents, model_name)
    vectorstore.save_local(VECTOR_DB_DIR)
    print(f"✅ Saved FAISS vectorstore to '{VECTOR_DB_DIR}'")

def load_vectorstore(embedder=None, backend: str = "faiss") -> FAISS:
    if not os.path.exists(VECTOR_DB_DIR):
        raise FileNotFoundError(f"No vectorstore found at '{VECTOR_DB_DIR}'. Run save_vectorstore() first.")
    if embedder is None:
        embedder = OllamaEmbeddings(model="nomic-embed-text")
    if backend.lower() == "faiss":
        vectorstore = FAISS.load_local(
            VECTOR_DB_DIR,
            embedder,
            allow_dangerous_deserialization=True
        )
    else:
        raise ValueError(f"Backend '{backend}' not supported")

    print(f"✅ Loaded {backend.upper()} vectorstore from '{VECTOR_DB_DIR}'")
    return vectorstore


