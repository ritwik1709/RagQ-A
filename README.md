# LangChain RAG Tutorial

A Retrieval-Augmented Generation (RAG) pipeline that lets you ask natural language questions about your documents and get AI-generated answers with source citations. Runs **100% locally** with no API costs.

## Tech Stack

- **LangChain** — Orchestrates the RAG pipeline
- **ChromaDB** — Vector database for storing document embeddings
- **HuggingFace** (all-MiniLM-L6-v2) — Free, local text embeddings
- **Ollama** (LLaMA 3.2) — Free, local LLM for answer generation

## Setup

### 1. Install Ollama

Download and install from [ollama.com/download](https://ollama.com/download), then pull the model:

```bash
ollama pull llama3.2
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 3. Install Markdown Dependencies

```bash
pip install "unstructured[md]"
```

## Usage

### Create the Database

Load your documents, split them into chunks, generate embeddings, and store in ChromaDB:

```bash
python create_database.py
```

### Query the Database

Ask questions about your documents:

```bash
python query_data.py "How does Alice meet the Mad Hatter?"
```

### Compare Embeddings

Test embedding similarity between words:

```bash
python compare_embeddings.py
```

## Using Your Own Data

1. Place your `.md` files in the `data/books/` directory
2. Run `python create_database.py` to rebuild the vector database
3. Query with `python query_data.py "your question here"`

## Project Structure

```
├── create_database.py    # Loads, chunks, and embeds documents into ChromaDB
├── query_data.py         # Queries the vector DB and generates answers via Ollama
├── compare_embeddings.py # Utility to compare word embeddings
├── test_load.py          # Test script to verify document loading
├── data/books/           # Source documents (Markdown files)
├── chroma/               # Vector database (auto-generated, gitignored)
└── requirements.txt      # Python dependencies
```

Based on the tutorial: [RAG+Langchain Python Project: Easy AI/Chat For Your Docs](https://www.youtube.com/watch?v=tcqEUSNCn8I&ab_channel=pixegami).
