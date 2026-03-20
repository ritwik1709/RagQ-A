# Document Q&A — RAG Application

Ask natural language questions about your documents and get AI-generated answers with source citations. Runs **100% locally** — no API keys or costs.

## Tech Stack

| Layer | Technology |
|---|---|
| Backend API | FastAPI + Uvicorn |
| RAG Framework | LangChain |
| Vector Database | ChromaDB |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` (local, free) |
| LLM | Ollama `llama3.2` (local, free) |
| Frontend | HTML / CSS / JavaScript |

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/download) installed

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/ritwik1709/langchain-rag-tutorial.git
cd langchain-rag-tutorial
```

### 2. Create and activate a virtual environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Mac/Linux
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Pull the Ollama model

```bash
ollama pull llama3.2
```

> First-time setup will also download the HuggingFace embedding model (~90MB) automatically on first run.

## Running the Web App

**Windows:**
```bash
run.bat
```

**Mac/Linux:**
```bash
bash run.sh
```

**Or manually:**
```bash
.venv/Scripts/python -m uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Then open **http://localhost:8000** in your browser.

## Using the Web App

1. **Upload** — drag and drop or click to upload a `.txt` or `.md` file
2. **Ask** — type your question and press Enter or click "Search Document"
3. **Results** — see the AI-generated answer, relevance scores, and source file

## CLI Usage (Alternative)

You can also run the pipeline directly from the terminal.

**Build the database from files in `data/books/`:**
```bash
python create_database.py
```

**Query the database:**
```bash
python query_data.py "How does Alice meet the Mad Hatter?"
```

**Compare word embeddings:**
```bash
python compare_embeddings.py
```

## Using Your Own Documents

1. Place your `.txt` or `.md` files in `data/books/`
2. Run `python create_database.py` to rebuild the vector database
3. Query via the web app or CLI

## Project Structure

```
├── app.py                 # FastAPI backend (REST API + serves frontend)
├── index.html             # Frontend UI
├── create_database.py     # CLI — build vector DB from documents
├── query_data.py          # CLI — query the vector DB
├── compare_embeddings.py  # Utility — compare word embedding similarity
├── test_load.py           # Utility — verify document loading works
├── run.bat                # Windows startup script
├── run.sh                 # Mac/Linux startup script
├── requirements.txt       # Python dependencies
├── data/books/            # Source documents (.txt / .md)
├── uploads/               # Documents uploaded via web UI
└── chroma/                # Vector database (auto-generated, gitignored)
```

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serves the web UI |
| `GET` | `/health` | Health check |
| `GET` | `/status` | Check if DB is ready |
| `POST` | `/upload` | Upload and process a document |
| `POST` | `/query` | Ask a question about the document |

Full interactive docs: **http://localhost:8000/docs**

## Troubleshooting

| Problem | Fix |
|---|---|
| `ollama: command not found` | Restart terminal after installing Ollama |
| `Connection refused` on query | Run `ollama serve` in a separate terminal |
| Slow first run | HuggingFace model downloading (~90MB, one-time) |
| `Please upload a document first` | Upload a file via the UI before querying |
| Module import error | Run `pip install -r requirements.txt` |

---

Based on: [RAG+Langchain Python Project: Easy AI/Chat For Your Docs](https://www.youtube.com/watch?v=tcqEUSNCn8I&ab_channel=pixegami)
