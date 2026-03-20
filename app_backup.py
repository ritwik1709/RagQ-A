from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os
import shutil
import traceback
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(title="Document Q&A API", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CHROMA_PATH = "chroma"
UPLOAD_FOLDER = "uploads"

# Create folders
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CHROMA_PATH, exist_ok=True)

print(f"✓ Upload folder: {os.path.abspath(UPLOAD_FOLDER)}")
print(f"✓ Database path: {os.path.abspath(CHROMA_PATH)}")

# Global variable for embedding function (lazy loaded)
embedding_function = None

def get_embedding_function():
    """Lazy load embedding function on first use"""
    global embedding_function
    if embedding_function is None:
        print("Loading HuggingFace embeddings model (first time, this may take a minute)...")
        from langchain_huggingface import HuggingFaceEmbeddings
        embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        print("✓ Embedding model loaded successfully")
    return embedding_function

# Models
class QueryRequest(BaseModel):
    question: str

# Endpoints
@app.get("/", response_class=JSONResponse)
def health_check():
    """Health check endpoint"""
    return {"status": "ok", "message": "Document Q&A API is running"}

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload and process document"""
    try:
        from langchain_community.document_loaders import TextLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_chroma import Chroma
        
        print(f"\n📤 Uploading file: {file.filename}")
        
        # Save file
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        print(f"✓ File saved to: {file_path}")
        
        # Load document
        print("Loading document...")
        loader = TextLoader(file_path, encoding='utf-8')
        documents = loader.load()
        print(f"✓ Document loaded: {len(documents)} document(s)")
        
        # Split into chunks
        print("Splitting text into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=100,
            length_function=len,
            add_start_index=True,
        )
        chunks = text_splitter.split_documents(documents)
        print(f"✓ Created {len(chunks)} chunks")
        
        # Clear existing database
        if os.path.exists(CHROMA_PATH):
            print("Clearing existing database...")
            shutil.rmtree(CHROMA_PATH)
            os.makedirs(CHROMA_PATH, exist_ok=True)
        
        # Create embeddings and store
        print("Creating embeddings (this may take a minute)...")
        db = Chroma.from_documents(
            chunks, 
            get_embedding_function(), 
            persist_directory=CHROMA_PATH
        )
        print("✓ Database created successfully")
        
        response = {
            "message": "Document processed successfully",
            "chunks": len(chunks)
        }
        print(f"✓ Upload complete: {response['message']}")
        return response
    
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/query", response_class=JSONResponse)
async def query_document(request: QueryRequest):
    """Query the document"""
    try:
        from langchain_chroma import Chroma
        
        print(f"\n❓ Query: {request.question}")
        
        # Check if database exists
        if not os.path.exists(CHROMA_PATH) or not os.listdir(CHROMA_PATH):
            raise HTTPException(status_code=400, detail="No document uploaded yet. Please upload a document first.")
        
        # Load database
        print("Loading database...")
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_function())
        
        # Search
        print("Searching database...")
        results = db.similarity_search_with_relevance_scores(request.question, k=3)
        
        if len(results) == 0:
            raise HTTPException(status_code=404, detail="No relevant information found in the document.")
        
        # Extract results
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        sources = [doc.metadata.get("source", "Unknown") for doc, _score in results]
        relevance_scores = [float(score) for doc, score in results]
        
        # Generate answer
        answer = f"Based on the document, here's what I found:\n\n{context_text}"
        
        response = {
            "question": request.question,
            "answer": answer,
            "sources": sources,
            "relevance_scores": relevance_scores
        }
        print(f"✓ Query complete. Found {len(results)} relevant sections")
        return response
    
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status", response_class=JSONResponse)
def get_status():
    """Check database status"""
    db_ready = os.path.exists(CHROMA_PATH) and len(os.listdir(CHROMA_PATH)) > 0
    return {"database_ready": db_ready}

if __name__ == "__main__":
    import uvicorn
    print("\n🚀 Starting Document Q&A API...")
    print("📍 API: http://localhost:8000")
    print("📍 Docs: http://localhost:8000/docs")
    print("📋 Health Check: http://localhost:8000/\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
