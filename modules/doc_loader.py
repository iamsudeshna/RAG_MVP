
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
from langchain.schema import Document


def load_and_split(path: str, chunk_size: int = 350, chunk_overlap: int = 35) -> List[Document]:
    
    loader = TextLoader(path, encoding="utf-8")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", "", "-"]
    )
    splits = splitter.split_documents(docs)

    print(f"Loaded {len(docs)} source docs -> created {len(splits)} chunks")
    if len(splits) > 0:
        print("Sample chunk (first 200 chars):")
        print(splits[0].page_content[:250])
    return splits


