import chromadb
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

# creates a local, persistent database on disk (no server, no API key needed)
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# a "collection" is like a table — one per document set/use case
collection = chroma_client.get_or_create_collection(name="rag_demo")

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
# def simple_chunk(text, chunk_size=300, overlap=50):
#     """Splits text into overlapping chunks by character count (simplest possible strategy)."""
#     text = text.strip()
#     chunks = []
#     start = 0
#     while start < len(text):
#         end = start + chunk_size
#         chunks.append(text[start:end])
#         start += chunk_size - overlap  # overlap prevents losing context at chunk boundaries
#     return chunks

# chunks = simple_chunk(sample_doc)
import re

def sentence_aware_chunk(text, max_chunk_size=300, overlap_sentences=1):
    """Splits text into sentences first, then groups sentences into chunks
    without ever cutting a sentence in half."""
    text = text.strip()
    # naive sentence splitter — splits on '. ', '! ', '? ' followed by a capital letter
    # (good enough for now; real systems use a proper sentence tokenizer — nltk/spacy)
    sentences = re.split(r'(?<=[.!?])\s+', text)

    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        if current_length + len(sentence) > max_chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            # keep last N sentences for overlap, so context carries across chunk boundaries
            current_chunk = current_chunk[-overlap_sentences:]
            current_length = sum(len(s) for s in current_chunk)

        current_chunk.append(sentence)
        current_length += len(sentence)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

chunks_v2 = sentence_aware_chunk(sample_doc)
for i, c in enumerate(chunks_v2):
    print(f"--- Chunk {i} ---\n{c}\n")
# Chroma can embed for you automatically, but let's stay explicit and reuse
# our own sentence-transformers model, so you know exactly what's happening
# chunk_embeddings = model.encode(chunks_v2).tolist()  # Chroma wants plain lists, not numpy arrays

# collection.add(
#     documents=chunks_v2,                          # the actual text
#     embeddings=chunk_embeddings,                # the vectors we computed
#     ids=[f"chunk_{i}" for i in range(len(chunks_v2))]  # unique ID per chunk — required
# )

# 

# results = collection.query(
#     query_embeddings=query_embedding,
#     n_results=2
# )

# for doc, distance in zip(results["documents"][0], results["distances"][0]):
#     print(f"[distance: {distance:.3f}] {doc}\n")
chunk_embeddings_v2 = model.encode(chunks_v2).tolist()

collection_v2 = chroma_client.get_or_create_collection(name="rag_demo_v2")
collection_v2.add(
    documents=chunks_v2,
    embeddings=chunk_embeddings_v2,
    ids=[f"chunk_{i}" for i in range(len(chunks_v2))]
)
query = "What determines good chunk size?"
# results_v2 = collection_v2.query(
#     query_embeddings=model.encode([query]).tolist(),
#     n_results=2
# )

# for doc, distance in zip(results_v2["documents"][0], results_v2["distances"][0]):
#     print(f"[distance: {distance:.3f}] {doc}\n")
# context = "\n\n".join(results_v2["documents"][0])

# prompt = f"""Answer the question using ONLY the context below. If the context doesn't contain the answer, say so.

# Context:
# {context}

# Question: {query}"""

# response = client.chat.completions.create(
#     model="openai/gpt-oss-120b",
#     messages=[{"role": "user", "content": prompt}]
# )

# print(response.choices[0].message.content)
DISTANCE_THRESHOLD = 1.6  # tune this per embedding model + dataset — there's no universal number

def rag_query(query, collection, threshold=DISTANCE_THRESHOLD, n_results=2):
    query_embedding = model.encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=n_results)

    docs = results["documents"][0]
    distances = results["distances"][0]

    best_distance = distances[0]  # results come back sorted, closest first

    if best_distance > threshold:
        return "No relevant information found in the knowledge base.", None

    context = "\n\n".join(docs)
    prompt = f"""Answer the question using ONLY the context below.

Context:
{context}

Question: {query}"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content, best_distance

# test both cases
answer, dist = rag_query("What determines good chunk size?", collection_v2)
print(f"[best distance: {dist}]\n{answer}\n")

answer2, dist2 = rag_query("What is the capital of Japan?", collection_v2)
print(f"[best distance: {dist2 if dist2 else 'N/A — blocked before generation'}]\n{answer2}\n")