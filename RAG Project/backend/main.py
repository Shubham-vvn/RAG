import os
import io
import csv
import json
import requests
from typing import List, Dict, Any
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import docx
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

app = FastAPI(title="Ask My Documents API", version="2.0.0")

# Enable CORS for the Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify the exact frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to hold RAG instances
embedding_model = None
vector_store = None

# Track indexed sources in-memory: source_name -> info dict
indexed_sources = {}

# Request models
class QuestionRequest(BaseModel):
    question: str

class LinkRequest(BaseModel):
    url: str

@app.on_event("startup")
def load_models():
    global embedding_model
    try:
        print("Loading BGE embedding model...")
        embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        print("BGE model loaded successfully.")
    except Exception as e:
        print(f"Error loading embedding model: {e}")

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    global vector_store, indexed_sources
    filename = file.filename
    lower_filename = filename.lower()
    
    supported_extensions = ['.pdf', '.docx', '.txt', '.md', '.csv']
    if not any(lower_filename.endswith(ext) for ext in supported_extensions):
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file format. Supported formats: {', '.join(supported_extensions)}"
        )

    try:
        file_bytes = await file.read()
        extracted_text = ""
        page_count = None

        # Parse specific formats
        if lower_filename.endswith('.pdf'):
            reader = PdfReader(io.BytesIO(file_bytes))
            page_count = len(reader.pages)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    extracted_text += page_text + "\n"
        
        elif lower_filename.endswith('.docx'):
            doc = docx.Document(io.BytesIO(file_bytes))
            extracted_text = "\n".join([p.text for p in doc.paragraphs if p.text])
            
        elif lower_filename.endswith('.csv'):
            decoded_str = file_bytes.decode("utf-8", errors="ignore")
            csv_reader = csv.reader(io.StringIO(decoded_str))
            rows = list(csv_reader)
            if rows:
                header = rows[0]
                row_texts = []
                for i, row in enumerate(rows[1:], start=1):
                    row_data = ", ".join([
                        f"{header[j]}={val}" if j < len(header) else f"col{j}={val}" 
                        for j, val in enumerate(row)
                    ])
                    row_texts.append(f"Row {i}: {row_data}")
                extracted_text = "\n".join(row_texts)
            else:
                extracted_text = ""
                
        elif lower_filename.endswith('.txt') or lower_filename.endswith('.md'):
            extracted_text = file_bytes.decode("utf-8", errors="ignore")

        if not extracted_text.strip():
            raise HTTPException(status_code=400, detail="The file contains no extractable text.")

        char_count = len(extracted_text)

        # --- Stage 4: Chunking ---
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        chunks = text_splitter.split_text(extracted_text)
        chunk_count = len(chunks)

        if chunk_count == 0:
            raise HTTPException(status_code=400, detail="The file is too short to index.")

        # --- Stage 5 & 6: Vector Ingestion (Cumulative) ---
        metadatas = [
            {"source": filename, "chunk_index": i} 
            for i in range(chunk_count)
        ]
        
        print(f"Indexing {chunk_count} chunks for '{filename}' in ChromaDB...")
        if vector_store is None:
            vector_store = Chroma.from_texts(
                texts=chunks,
                embedding=embedding_model,
                metadatas=metadatas
            )
        else:
            vector_store.add_texts(
                texts=chunks,
                metadatas=metadatas
            )

        # Update global list
        indexed_sources[filename] = {
            "type": "file",
            "char_count": char_count,
            "chunk_count": chunk_count,
            "pages": page_count
        }

        return {
            "status": "success",
            "filename": filename,
            "character_count": char_count,
            "page_count": page_count,
            "chunk_count": chunk_count
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process file '{filename}': {str(e)}")

@app.post("/upload-link")
async def upload_link(request: LinkRequest):
    global vector_store, indexed_sources
    url = request.url.strip()
    
    if not url.startswith("http://") and not url.startswith("https://"):
        raise HTTPException(status_code=400, detail="Invalid URL. It must begin with http:// or https://")

    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        print(f"Fetching URL: {url}")
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail=f"Failed to fetch website (HTTP Status {response.status_code})")

        soup = BeautifulSoup(response.text, "html.parser")
        
        # Decompose elements we don't want to parse
        for element in soup(["script", "style", "nav", "header", "footer", "aside"]):
            element.decompose()
            
        extracted_text = soup.get_text(separator="\n")
        lines = [line.strip() for line in extracted_text.splitlines() if line.strip()]
        extracted_text = "\n".join(lines)

        if not extracted_text.strip():
            raise HTTPException(status_code=400, detail="No readable text could be extracted from this URL.")

        char_count = len(extracted_text)

        # Chunking
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        chunks = text_splitter.split_text(extracted_text)
        chunk_count = len(chunks)

        if chunk_count == 0:
            raise HTTPException(status_code=400, detail="Extracted text is too short to index.")

        # Ingest to Chroma
        metadatas = [
            {"source": url, "chunk_index": i} 
            for i in range(chunk_count)
        ]
        
        print(f"Indexing {chunk_count} chunks for website '{url}' in ChromaDB...")
        if vector_store is None:
            vector_store = Chroma.from_texts(
                texts=chunks,
                embedding=embedding_model,
                metadatas=metadatas
            )
        else:
            vector_store.add_texts(
                texts=chunks,
                metadatas=metadatas
            )

        indexed_sources[url] = {
            "type": "link",
            "char_count": char_count,
            "chunk_count": chunk_count,
            "pages": None
        }

        return {
            "status": "success",
            "filename": url,
            "character_count": char_count,
            "chunk_count": chunk_count
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to ingest URL '{url}': {str(e)}")

@app.get("/sources")
async def get_sources():
    return {"sources": indexed_sources}

@app.post("/clear")
async def clear_database():
    global vector_store, indexed_sources
    vector_store = None
    indexed_sources.clear()
    return {"status": "success", "detail": "Vector index and sources dashboard reset."}

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    global vector_store, embedding_model, indexed_sources
    if vector_store is None:
        raise HTTPException(status_code=400, detail="No content sources have been indexed yet.")

    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        # Retrieve candidate matches
        retrieved_results = vector_store.similarity_search_with_score(question, k=3)
        
        # Build map of (source, chunk_index) -> Chroma DB ID
        all_data = vector_store.get()
        id_map = {}
        if all_data and "ids" in all_data:
            for doc_id, meta in zip(all_data["ids"], all_data["metadatas"]):
                if meta and "source" in meta and "chunk_index" in meta:
                    id_map[(meta["source"], meta["chunk_index"])] = doc_id

        # Format candidates
        chunks_response = []
        for idx, (doc, score) in enumerate(retrieved_results):
            source = doc.metadata.get("source", "Unknown")
            chunk_idx = doc.metadata.get("chunk_index", "Unknown")
            doc_id = id_map.get((source, chunk_idx), "Unknown ID")
            
            words_in_common = [
                w for w in question.lower().split() 
                if len(w) > 3 and w in doc.page_content.lower()
            ]
            common_str = ", ".join([f"'{w}'" for w in words_in_common]) if words_in_common else "semantic context"
            
            explanation = (
                f"This chunk was retrieved with a similarity distance score of {score:.4f}. "
                f"It was selected because it contains keywords or semantic concepts related to your query "
                f"(matching terms: {common_str})."
            )
            
            chunks_response.append({
                "index": idx + 1,
                "id": doc_id,
                "text": doc.page_content,
                "score": float(score),
                "explanation": explanation,
                "metadata": doc.metadata
            })

        # LLM prompt composition
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key or api_key == "your_key_here":
            raise HTTPException(status_code=500, detail="GROQ_API_KEY is not configured on the backend server.")

        context_text = "\n\n".join([
            f"[Source: {c['metadata'].get('source', 'Unknown Document')}, Chunk {c['metadata'].get('chunk_index', 'N/A')}] {c['text']}" 
            for c in chunks_response
        ])
        
        messages = [
            (
                "system",
                "You are a helpful assistant. Answer the user's question based ONLY on the provided context. "
                "Each context block starts with [Source: filename]. Feel free to mention which source document "
                "or website you are using in your answer. "
                "If the context does not contain the answer, say 'I don't have enough information to answer this question.' "
                "Do not make up answers or use external knowledge."
            ),
            (
                "user",
                f"Context:\n{context_text}\n\nQuestion: {question}"
            )
        ]

        llm = ChatGroq(
            api_key=api_key,
            model_name="llama-3.1-8b-instant",
            temperature=0.0
        )
        
        response = llm.invoke(messages)
        
        return {
            "answer": response.content,
            "retrieved_chunks": chunks_response
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")
