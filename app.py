from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os
import shutil
import traceback

app = FastAPI(title="Document Q&A API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CHROMA_PATH = "chroma"
UPLOAD_FOLDER = "uploads"
LLM_MODE = os.getenv("LLM_MODE", "extractive").strip().lower()

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CHROMA_PATH, exist_ok=True)

embedding_function = None

def get_embedding_function():
    global embedding_function
    if embedding_function is None:
        print("Loading HuggingFace embeddings...")
        from langchain_huggingface import HuggingFaceEmbeddings
        embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        print("✓ Embeddings loaded")
    return embedding_function

class QueryRequest(BaseModel):
    question: str

@app.get("/")
def serve_frontend():
    return FileResponse("index.html")

@app.get("/health")
def health_check():
    return {"status": "ok", "message": "API running"}

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    try:
        from langchain_community.document_loaders import TextLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_chroma import Chroma

        # Validate file type
        allowed_extensions = {'.txt', '.md'}
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type '{file_ext}'. Please upload a .txt or .md file."
            )

        print(f"Uploading {file.filename}...")

        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)

        loader = TextLoader(file_path, encoding='utf-8')
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=100,
            length_function=len,
            add_start_index=True,
        )
        chunks = text_splitter.split_documents(documents)

        if not chunks:
            raise HTTPException(status_code=400, detail="Document is empty or could not be split into chunks.")

        # Rebuild chroma DB from scratch
        if os.path.exists(CHROMA_PATH):
            shutil.rmtree(CHROMA_PATH)

        print(f"Creating embeddings for {len(chunks)} chunks...")
        db = Chroma.from_documents(chunks, get_embedding_function(), persist_directory=CHROMA_PATH)
        print(f"✓ Saved {len(chunks)} chunks to ChromaDB")

        return {"message": "Document processed successfully", "chunks": len(chunks), "filename": file.filename}

    except HTTPException:
        raise
    except Exception as e:
        print(f"Upload error: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def build_answer(question: str, context_text: str) -> str:
    """Generate an answer using configured mode; fallback to extractive for cloud deploys."""
    if LLM_MODE == "ollama":
        try:
            from langchain_ollama import ChatOllama
            from langchain_core.prompts import ChatPromptTemplate

            prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
            prompt = prompt_template.format(context=context_text, question=question)
            model = ChatOllama(model="llama3.2")
            return model.invoke(prompt).content
        except Exception as e:
            print(f"Ollama unavailable, falling back to extractive mode: {e}")

    # Deployment-safe fallback: return concise answer derived from retrieved context.
    top_chunk = context_text.split("\n\n---\n\n")[0].strip()
    if len(top_chunk) > 1200:
        top_chunk = top_chunk[:1200] + "..."
    return (
        "Answer (extractive mode):\n"
        f"Based on the document, the most relevant information for '{question}' is:\n\n"
        f"{top_chunk}"
    )

@app.post("/query")
async def query_document(request: QueryRequest):
    try:
        from langchain_chroma import Chroma

        print(f"Query: {request.question}")

        if not os.path.exists(CHROMA_PATH) or not os.listdir(CHROMA_PATH):
            raise HTTPException(status_code=400, detail="Please upload a document first before querying.")

        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_function())
        results = db.similarity_search_with_relevance_scores(request.question, k=3)

        if not results or results[0][1] < 0.3:
            raise HTTPException(status_code=404, detail="No relevant information found in the document for your question.")

        context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
        sources = list(set([doc.metadata.get("source", "Unknown") for doc, _ in results]))
        relevance_scores = [round(float(score), 4) for _, score in results]

        answer = build_answer(request.question, context_text)
        print(f"✓ Answer generated with mode: {LLM_MODE}")

        return {
            "question": request.question,
            "answer": answer,
            "sources": sources,
            "relevance_scores": relevance_scores,
            "context_used": context_text
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"Query error: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

@app.get("/status")
def get_status():
    db_ready = os.path.exists(CHROMA_PATH) and len(os.listdir(CHROMA_PATH)) > 0
    return {"database_ready": db_ready}

if __name__ == "__main__":
    import uvicorn
    print("\n🚀 Starting server at http://localhost:8000")
    print("🌐 Open http://localhost:8000 in your browser\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
