import os
import google.generativeai as genai
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# my api key
genai.configure(api_key="AIzaSyAO6GOdKe-gZQlWOUwZZiKFNstF8xFVKjU")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  

# loading pdfs
pdfs = ["Attention is all you need.pdf", "BERT.pdf", "GPT-3.pdf", "Language-Image.pdf", "LLamA.pdf"]
documents = []

# extracting text from pdfs
for pdf in pdfs:
    reader = PdfReader(pdf)
    for page in reader.pages:
        text = page.extract_text()
        if text:
            documents.append(text)

# making chunks of 1000size
def split_text(text, chunk_size=1000, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

chunks = []
for doc in documents:
    chunks.extend(split_text(doc))



# chunks to vectors
embeddings = embedding_model.encode(chunks, convert_to_numpy=True)

# checking dimension
dimension = embeddings.shape[1]
# creating faiss index
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)


def answer_question(question, top_k=3):
    q_embedding = embedding_model.encode([question])[0]
    distances, indices = index.search(np.array([q_embedding]), top_k)

    relevant_chunks = [chunks[i] for i in indices[0]]
    context = "\n\n".join(relevant_chunks)

    prompt = f"""Answer the following question based only on the context below.

Context:
{context}

Question:
{question}

Answer:"""

    model = genai.GenerativeModel("models/gemini-2.5-flash-preview-05-20")

    response = model.generate_content(prompt)
    return response.text


questions = [
    "What is the main innovation introduced in the 'Attention is All You Need' paper?",
    "How does BERT differ from traditional left-to-right language models?",
    "Describe the few-shot learning capability of GPT-3 with an example.",
    "What is the loss function used in CLIP and why is it effective?",
    "What approach does LLaMA take to reduce computational cost during training?"
]

for q in questions:
    ans = answer_question(q)
    print(f"\n Question: {q}\n Answer: {ans}")
