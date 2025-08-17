
from typing import Optional
from langchain_community.embeddings import OllamaEmbeddings
#from langchain_ollama import OllamaEmbeddings


def get_embedder(
    model_name: str = "nomic-embed-text",
    base_url: Optional[str] = None) -> OllamaEmbeddings:
    
    kwargs = {"model": model_name}
    if base_url:
        kwargs["base_url"] = base_url
    return OllamaEmbeddings(**kwargs)
