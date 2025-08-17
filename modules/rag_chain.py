
from typing import Optional
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


_DEFAULT_PROMPT = """You are a helpful assistant. Use the provided context to answer the question.
If the answer cannot be found in the context, say you don't know.

Context:
{context}

Question:
{question}
"""


def make_rag_chain(retriever, llm, prompt_template: Optional[str] = None):
    template = prompt_template or _DEFAULT_PROMPT
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=template
    )
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,  
    )
    return chain
