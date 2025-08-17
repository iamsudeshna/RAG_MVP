## This Readme file helps to :
a. Draw the project file structure.
b. Explain why the structure is production-level.
c. Mention Ollama embeddings, FAISS, LLaMA2, LangChain, FastAPI.

This is a student-friendly, production-level RAG application built with LangChain, FAISS, Ollama embeddings, LLaMA2, and FastAPI.
All tools and frameworks used are open-source.

This folder design is inspired by production-level ML/AI systems:
✅ Separation of concerns → loaders, embeddings, vector DB, and chains are all modular.
✅ Scalability → easy to extend (swap Ollama embeddings with OpenAI, or FAISS with Pinecone).
✅ Maintainability → changes in one component (say vector DB) don’t break others.
✅ Reusability → Each module is a clean building block for future projects.
This means anyone can start small, but the same structure works if scaled into production.

## The below mentioned is the project folder structure used:

RAG_MVP/
│── data/                     # Raw documents (knowledge base for RAG)
│── db/                       # Database-related files
│── env/                      # Environment configs (e.g., virtual environment)
│── modules/                  # Core logic (modularized for reusability & clarity)
│   │── doc_loader.py         # Loads and splits documents into chunks
│   │── embedder.py           # Generates embeddings (Ollama embeddings)
│   │── llm_wrapper.py        # Wraps the LLaMA2 LLM for inference
│   │── rag_chain.py          # Defines the RAG pipeline (Retriever + LLM)
│   │── vectordb.py           # Handles FAISS vector store (insert, search)
│── vectorstore/              # Persistent FAISS index storage
│── .env                      # Environment variables (API keys, configs)
│── client.py                 # Simple client to query the RAG system
│── main.py                   # FastAPI server exposing endpoints
│── requirements.txt          # Python dependencies

## Tech Stack :

LangChain → Orchestration framework
LLaMA2 → Open-source LLM used for generation
Ollama embeddings → To convert text into vectors (We can use 'sentence-transformers/all-MiniLM-L6-v2' embedding model, this is also opensource from HuggingFace)
FAISS → Vector database with similarity search   (We can use ChromaDB as well)
FastAPI → REST API layer
----------- All components are open-source

FAISS is a vector database optimized for fast similarity search.
When you query, the text is embedded using Ollama embeddings → this becomes a vector.
FAISS computes similarity (cosine or inner product distance) between this query vector and stored document vectors.
The most relevant chunks are returned as context for the LLM.
In our RAG app:
vectordb.py manages FAISS → rag_chain.py calls the retriever → results go to the LLM.

In LangChain, chains = sequences of components that execute in a specific order.
Some key retriever-related chains:

LLMChain → Prompt → LLM (basic chain)
SQLDatabaseChain → Natural language → SQL query → DB execution
RetrievalQA → Query → Retriever → Context → LLM (most common RAG)
ConversationalRetrievalChain → Same as above, but with chat history
Summarization Chains →
StuffDocumentsChain (all docs stuffed into one prompt)
MapReduceDocumentsChain (summarize pieces, then combine)
RefineDocumentsChain (iteratively improve summary)

In this project, we used RetrievalQA with chain_type="stuff" →
All retrieved docs are stuffed into the prompt → passed to LLaMA2 → final answer.

## How to run? 
### 1. Clone repo
a.  git clone https://github.com/iamsudeshna/<repo_name>.git
b.  cd <repo_name>

### 2. Create environment & install deps
a. python -m venv env
b. source env/bin/activate  (in MacOS)
c. pip install -r requirements.txt

### 3. Run FastAPI server
uvicorn main:app --reload --host 0.0.0.0 --port 8001 --reload 

### 4. Query via client
a. Change the question according to what is needed "<question>?"
b. python client.py

## Workflow
                
         doc_loader → embedder → vectordb → retriever → llm → FastAPI
