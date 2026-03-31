# 🎓 STEM Teacher Assistant — RAG Chatbot

A RAG-powered chatbot that answers teacher questions strictly 
from uploaded STEM curriculum documents.
Built with Mistral 7B running locally via Ollama.

## Tech Stack
- Python + Flask
- Mistral 7B (via Ollama — runs 100% locally)
- ChromaDB (vector database)
- LangChain
- Sentence Transformers

## Features
- Upload STEM PDF documents
- Ask teaching questions in natural language
- Answers strictly from uploaded documents
- Shows source page references
- Runs fully offline — no API costs

## How to Run
1. Install Ollama and pull Mistral: `ollama pull mistral`
2. Install dependencies: `pip install -r requirements.txt`
3. Run: `python app.py`
4. Open: `http://127.0.0.1:5001`

## Architecture
PDF Upload → PyMuPDF → Text Chunks → 
ChromaDB Vectors → Mistral 7B → Answer + Sources