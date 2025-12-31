"""
RAG Server - Retrieval-Augmented Generation Server
Handles document loading and querying using VectorDB and OpenAI
"""

import os
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from llm import LLM
from query_translation.multi_query import MultiQuery
from query_translation.rag_fusion import RagFusion

# Load environment variables
load_dotenv()

app = FastAPI(title="RAG Server", description="Retrieval-Augmented Generation Server")

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OpenAI API key must be provided via parameter or OPENAI_API_KEY environment variable")

# Convert environment variable string to boolean
# Handles: "True", "true", "1", "yes", "on" -> True
#          "False", "false", "0", "no", "off", None, "" -> False
truthList = ("true", "1", "yes", "on")
multi_query_enabled = os.getenv("ENABLE_MULTI_QUERY", "False").lower() in truthList
rag_fusion_enabled = os.getenv("ENABLE_RAG_FUSION", "False").lower() in truthList

# Initialize OpenAI components
embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
llm = LLM.get_instance().get_chat_openai()

# Vector store instance
vector_store: Optional[Chroma] = None
PERSIST_DIRECTORY = "./chroma_db"

# Request/Response models
class LoadDocumentRequest(BaseModel):
    file_path: str
    chunk_size: int = 1000
    chunk_overlap: int = 200

class QueryRequest(BaseModel):
    query: str
    k: int = 4  # Number of documents to retrieve

class QueryResponse(BaseModel):
    answer: str
    source_documents: List[str]

class StatusResponse(BaseModel):
    status: str
    message: str

def get_vector_store() -> Chroma:
    """Get or create the vector store"""
    global vector_store
    if vector_store is None:
        # Create a new vector store or load existing one
        if os.path.exists(PERSIST_DIRECTORY):
            vector_store = Chroma(
                persist_directory=PERSIST_DIRECTORY,
                embedding_function=embeddings
            )
        else:
            # Create empty vector store
            vector_store = Chroma(
                persist_directory=PERSIST_DIRECTORY,
                embedding_function=embeddings
            )
    return vector_store

def load_document(file_path: str) -> List:
    """Load document based on file extension"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    file_ext = os.path.splitext(file_path)[1].lower()
    
    if file_ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif file_ext in [".txt", ".md"]:
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file type: {file_ext}")
    
    return loader.load()

@app.get("/", response_model=StatusResponse)
async def root():
    """Health check endpoint"""
    return StatusResponse(
        status="running",
        message="RAG Server is running. Use /docs for API documentation."
    )

@app.post("/load-document", response_model=StatusResponse)
async def load_document_endpoint(request: LoadDocumentRequest):
    """Load a document into the VectorDB"""
    try:
        # Load document
        documents = load_document(request.file_path)
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap
        )
        splits = text_splitter.split_documents(documents)
        
        # Get or create vector store
        vector_store = get_vector_store()
        
        # Add documents to vector store
        vector_store.add_documents(splits)
        
        return StatusResponse(
            status="success",
            message=f"Successfully loaded {len(splits)} document chunks from {request.file_path}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """Query the VectorDB and get an answer"""
    try:
        vector_store = get_vector_store()
        
        # Check if vector store has documents
        if vector_store._collection.count() == 0:
            raise HTTPException(
                status_code=400,
                detail="No documents loaded. Please load a document first using /load-document"
            )
        
        # Create retrieval chain
        retriever = vector_store.as_retriever(search_kwargs={"k": request.k})
        # Create the chain using LCEL
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        # Create prompt template
        prompt_template = """Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.

        {context}

        Question: {question}
        Answer:"""
        prompt = ChatPromptTemplate.from_template(prompt_template)

        # Modify the retriever to use multi query or rag fusion
        if multi_query_enabled:
            multi_query = MultiQuery()
            retriever = multi_query.generate_multi_queries() | retriever.map() | multi_query.get_unique_union
        
        
        if rag_fusion_enabled:
            multi_query = RagFusion()
            retriever = multi_query.generate_rag_fusion_queries() | retriever.map() | multi_query.reciprocal_rank_fusion
        

        retrieval_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        # Execute query
        answer = retrieval_chain.invoke(request.query)
        print("Answer: ", answer)
        # Get source documents
        retrieved_docs = retriever.invoke(request.query)
        print("Documents: ", retrieved_docs)
        source_docs = [doc.page_content for doc in retrieved_docs]
        
        return QueryResponse(
            answer=answer,
            source_documents=source_docs
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status", response_model=StatusResponse)
async def status():
    """Get server status and document count"""
    try:
        vector_store = get_vector_store()
        doc_count = vector_store._collection.count()
        return StatusResponse(
            status="ready",
            message=f"VectorDB contains {doc_count} document chunks"
        )
    except Exception as e:
        return StatusResponse(
            status="error",
            message=str(e)
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005)

