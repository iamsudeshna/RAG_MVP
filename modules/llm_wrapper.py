
from typing import Optional
from langchain_community.llms import Ollama

def get_llm(
    model_name: str = "llama2",
    base_url: Optional[str] = None,
    temperature: float = 0.2,
    num_ctx: int = 4096,
):

    kwargs = {
        "model": model_name,
        "temperature": temperature,
        "num_ctx": num_ctx,
    }
    if base_url:
        kwargs["base_url"] = base_url
    return Ollama(**kwargs)



