from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer
import chromadb
import os
import json
from dotenv import load_dotenv
from groq import Groq
import re

load_dotenv()
client = Groq(api_key=os.environ["GROQ_API_KEY"])

model = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path="./chroma_db")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def sentence_aware_chunk(text, max_chunk_size=300, overlap_sentences=1):
    text = text.strip()
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = []
    current_length = 0
    for sentence in sentences:
        if current_length + len(sentence) > max_chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = current_chunk[-overlap_sentences:]
            current_length = sum(len(s) for s in current_chunk)
        current_chunk.append(sentence)
        current_length += len(sentence)
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

# --- the multi-document dataset (this is what the GIL query needs) ---
documents = {
    "doc_python_basics.txt": """
Python is a high-level, interpreted programming language known for its readability.
Variables in Python don't require explicit type declarations. Lists are ordered,
mutable collections, while tuples are ordered but immutable. Dictionaries store
key-value pairs and are widely used for structured data.
""",
    "doc_python_performance.txt": """
Python's performance can be a bottleneck for CPU-intensive tasks due to the Global
Interpreter Lock (GIL), which prevents true multi-threading for CPU-bound code.
For performance-critical sections, developers often use libraries like NumPy, which
run compiled C code under the hood, or switch to multiprocessing instead of threading.
""",
    "doc_js_basics.txt": """
JavaScript is a dynamically typed language primarily used for web development.
Variables can be declared with var, let, or const. Arrays in JavaScript are
similar to Python lists but have different built-in methods. Objects store
key-value pairs, similar to Python dictionaries but with different syntax.
""",
    "doc_rag_overview.txt": """
Retrieval-Augmented Generation (RAG) combines a retrieval system with a language model.
Instead of relying solely on parametric knowledge learned during training, RAG systems
fetch relevant documents at inference time.

The retrieval step typically uses vector similarity search. Documents are split into
chunks, embedded, and stored in a vector database like Chroma, Pinecone, or FAISS.

Chunk size matters significantly. Too small, and you lose context. Too large, and
irrelevant information dilutes the embedding, hurting retrieval accuracy. A common
starting point is 200-500 tokens per chunk with some overlap between chunks.
""",
    "doc_vector_db_comparison.txt": """
Choosing a vector database depends on your scale and infrastructure needs. Chroma is
lightweight and great for local development or small projects. Pinecone is fully
managed and scales well for production but has ongoing costs. FAISS is a library,
not a full database — extremely fast, but you handle persistence and infra yourself.
""",
}

all_chunks = []
all_metadatas = []
for doc_name, doc_text in documents.items():
    doc_chunks = sentence_aware_chunk(doc_text, max_chunk_size=300, overlap_sentences=1)
    for i, chunk in enumerate(doc_chunks):
        all_chunks.append(chunk)
        all_metadatas.append({"source": doc_name, "chunk_index": i})

print(f"Total chunks across all documents: {len(all_chunks)}")

chunk_embeddings_multi = model.encode(all_chunks).tolist()
collection_multi = chroma_client.get_or_create_collection(name="rag_multi_doc")
collection_multi.add(
    documents=all_chunks,
    embeddings=chunk_embeddings_multi,
    metadatas=all_metadatas,
    ids=[f"chunk_{i}" for i in range(len(all_chunks))]
)

# --- BM25 index, built on the SAME chunk set ---
tokenized_chunks = [chunk.lower().split() for chunk in all_chunks]
bm25 = BM25Okapi(tokenized_chunks)

def bm25_search(query, bm25, chunks, top_k=3):
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)  # one score per chunk

    # pair each chunk with its score, sort descending (higher = more relevant, like the reranker)
    ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]

def hybrid_search(query, collection, bm25, chunks, all_metadatas, top_k=3, alpha=0.5):
    """
    alpha controls the blend: alpha=1.0 is pure embedding search,
    alpha=0.0 is pure BM25. 0.5 is an even split.
    """
    # --- embedding side ---
    query_embedding = model.encode([query]).tolist()
    emb_results = collection.query(query_embeddings=query_embedding, n_results=len(chunks))
    emb_docs = emb_results["documents"][0]
    emb_distances = emb_results["distances"][0]

    # convert distance (lower=better) into a similarity-like score (higher=better), normalized 0-1
    max_dist = max(emb_distances) if emb_distances else 1
    emb_scores = {doc: 1 - (dist / max_dist) for doc, dist in zip(emb_docs, emb_distances)}

    # --- BM25 side ---
    tokenized_query = query.lower().split()
    bm25_raw_scores = bm25.get_scores(tokenized_query)
    max_bm25 = max(bm25_raw_scores) if max(bm25_raw_scores) > 0 else 1
    bm25_scores = {chunk: score / max_bm25 for chunk, score in zip(chunks, bm25_raw_scores)}  # normalize 0-1

    # --- combine ---
    combined = {}
    for chunk in chunks:
        e_score = emb_scores.get(chunk, 0)
        b_score = bm25_scores.get(chunk, 0)
        combined[chunk] = alpha * e_score + (1 - alpha) * b_score

    ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]

print("=== BM25 alone: exact-term query ===")
results_bm25 = bm25_search("Global Interpreter Lock GIL", bm25, all_chunks, top_k=3)
for chunk, score in results_bm25:
    print(f"[BM25 score: {score:.3f}] {chunk}\n")

print("=== BM25 alone: paraphrased query (should struggle) ===")
results_bm25_2 = bm25_search("why does python struggle with parallel processing", bm25, all_chunks, top_k=3)
for chunk, score in results_bm25_2:
    print(f"[BM25 score: {score:.3f}] {chunk}\n")

print("=== Hybrid: same paraphrased query ===")
results_hybrid = hybrid_search("why does python struggle with parallel processing", collection_multi, bm25, all_chunks, all_metadatas)
for chunk, score in results_hybrid:
    print(f"[hybrid score: {score:.3f}] {chunk}\n")