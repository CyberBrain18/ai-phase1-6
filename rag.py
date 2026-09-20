from sentence_transformers import SentenceTransformer
import numpy as np
import os
import json
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.environ["GROQ_API_KEY"])

# a small, fast, local embedding model — runs on your machine, no API needed
model = SentenceTransformer("all-MiniLM-L6-v2")

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))    # low, e.g. ~0.1 # (3, 384) — 3 sentences, each a 384-number vector

sample_doc = """
Retrieval-Augmented Generation (RAG) combines a retrieval system with a language model.
Instead of relying solely on parametric knowledge learned during training, RAG systems
fetch relevant documents at inference time.

The retrieval step typically uses vector similarity search. Documents are split into
chunks, embedded, and stored in a vector database like Chroma, Pinecone, or FAISS.

Chunk size matters significantly. Too small, and you lose context. Too large, and
irrelevant information dilutes the embedding, hurting retrieval accuracy. A common
starting point is 200-500 tokens per chunk with some overlap between chunks.
"""

def simple_chunk(text, chunk_size=300, overlap=50):
    """Splits text into overlapping chunks by character count (simplest possible strategy)."""
    text = text.strip()
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap  # overlap prevents losing context at chunk boundaries
    return chunks

chunks = simple_chunk(sample_doc)
for i, c in enumerate(chunks):
    print(f"--- Chunk {i} ---\n{c}\n")

# 1. Index: embed all chunks once, upfront
chunk_embeddings = model.encode(chunks)

def retrieve(query, chunk_embeddings, chunks, top_k=2):
    query_embedding = model.encode([query])[0]
    similarities = [cosine_similarity(query_embedding, ce) for ce in chunk_embeddings]
    top_indices = np.argsort(similarities)[::-1][:top_k]  # sort descending, take top_k
    return [(chunks[i], similarities[i]) for i in top_indices]

# 2. Retrieval
query = "What determines good chunk size?"
results = retrieve(query, chunk_embeddings, chunks)

for chunk_text, score in results:
    print(f"[score: {score:.3f}] {chunk_text}\n")
    
context = "\n\n".join([c for c, score in results])

prompt = f"""Answer the question using ONLY the context below. If the context doesn't contain the answer, say so.

Context:
{context}

Question: {query}"""

response = client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=[{"role": "user", "content": prompt}]
)

print(response.choices[0].message.content)
