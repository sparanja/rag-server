# RAG Server

A Retrieval-Augmented Generation (RAG) server built with Python, LangChain, OpenAI API, and ChromaDB vector database.

image.png

## Features

- Load documents (TXT, PDF) into a vector database
- Query documents using natural language
- Retrieve relevant document chunks based on semantic similarity
- Generate answers using OpenAI's GPT models with context from retrieved documents

## Prerequisites

- Python 3.8 or higher
- OpenAI API key

## Installation

1. Navigate to the project directory:
```bash
cd /Users/sumanthparanjape/Desktop/Unified/codebase/ai_ml_projects/rag_server
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your OpenAI API key:
```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

Or export it as an environment variable:
```bash
export OPENAI_API_KEY=your_api_key_here
```

## Usage

### Start the Server

```bash
python rag_server.py
```

Or using uvicorn directly:
```bash
uvicorn rag_server:app --reload --host 0.0.0.0 --port 8000
```

The server will start on `http://localhost:8000`

### API Documentation

Once the server is running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### Load a Document

**Using curl:**
```bash
curl -X POST "http://localhost:8000/load-document" \
  -H "Content-Type: application/json" \
  -d '{
    "file_path": "example_document.txt",
    "chunk_size": 1000,
    "chunk_overlap": 200
  }'
```

**Using Python:**
```python
import requests

response = requests.post(
    "http://localhost:8000/load-document",
    json={
        "file_path": "example_document.txt",
        "chunk_size": 1000,
        "chunk_overlap": 200
    }
)
print(response.json())
```

### Query the Documents

**Using curl:**
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is artificial intelligence?",
    "k": 4
  }'
```

**Using Python:**
```python
import requests

response = requests.post(
    "http://localhost:8000/query",
    json={
        "query": "What is artificial intelligence?",
        "k": 4
    }
)
result = response.json()
print("Answer:", result["answer"])
print("\nSource Documents:")
for doc in result["source_documents"]:
    print("-", doc[:100] + "...")
```

### Check Server Status

```bash
curl http://localhost:8000/status
```

## API Endpoints

### `GET /`
Health check endpoint

### `POST /load-document`
Load a document into the vector database

**Request Body:**
```json
{
  "file_path": "path/to/document.txt",
  "chunk_size": 1000,
  "chunk_overlap": 200
}
```

### `POST /query`
Query the vector database and get an answer

**Request Body:**
```json
{
  "query": "Your question here",
  "k": 4
}
```

**Response:**
```json
{
  "answer": "Generated answer based on retrieved documents",
  "source_documents": ["chunk 1", "chunk 2", ...]
}
```

### `GET /status`
Get server status and document count

## Project Structure

```
rag_server/
├── rag_server.py          # Main server application
├── requirements.txt       # Python dependencies
├── .env.example          # Environment variables template
├── example_document.txt  # Sample document for testing
├── README.md             # This file
└── chroma_db/            # Vector database storage (created automatically)
```

## How It Works

1. **Document Loading**: Documents are loaded and split into chunks using a text splitter
2. **Embedding**: Each chunk is converted to a vector embedding using OpenAI's embedding model
3. **Storage**: Embeddings are stored in ChromaDB vector database
4. **Query**: User queries are converted to embeddings and used to find similar document chunks
5. **Generation**: Retrieved chunks are used as context for OpenAI's GPT model to generate answers

## Supported File Types

- `.txt` - Plain text files
- `.md` - Markdown files
- `.pdf` - PDF files

## Configuration

You can modify the following in `rag_server.py`:
- `PERSIST_DIRECTORY`: Where to store the vector database
- `chunk_size` and `chunk_overlap`: Document splitting parameters
- OpenAI model selection (default: `gpt-3.5-turbo`)

## Notes

- The vector database persists to disk, so loaded documents remain available after server restart
- Make sure you have sufficient OpenAI API credits
- For production use, consider adding authentication, rate limiting, and error handling improvements

## Activate Python virtual environment:
source envRag/bin/activate
