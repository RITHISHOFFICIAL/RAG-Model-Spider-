RAG-QA System on ML Papers

A lightweight Retrieval-Augmented Generation (RAG) web app that answers user queries based on five key Machine Learning research papers using Google's Gemini API, FAISS for semantic search, and Flask for the interface.



Included Research Papers
Attention is All You Need – Transformer architecture.

BERT – Deep bidirectional transformers.

GPT-3 – Few-shot learning and autoregressive generation.

CLIP (Language-Image) – Vision-language contrastive pretraining.

LLaMA – Efficient open-source foundation models.



Features

Extracts and embeds text from PDF research papers.

Stores embeddings in a FAISS index for fast retrieval.

Accepts user questions via a Flask web app.

Uses top-k relevant chunks to form a prompt.

Sends the prompt to Gemini (via google.generativeai) for final answers.

Supports few-shot examples for context-rich querying.



.

├── app.py                 # Flask app

├── model.py               # Core RAG logic 


├── templates/
│   └── index.html         # Basic HTML form for question input

├── *.pdf                  # Research papers used for context



