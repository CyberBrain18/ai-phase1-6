import chromadb
from sentence_transformers import SentenceTransformer
import numpy as np
import os
import json
from dotenv import load_dotenv
from groq import Groq
import re

load_dotenv()
client = Groq(api_key=os.environ["GROQ_API_KEY"])

# a small, fast, local embedding model — runs on your machine, no API needed
model = SentenceTransformer("all-MiniLM-L6-v2")

chroma_client = chromadb.PersistentClient(path="./chroma_db")

# a "collection" is like a table — one per document set/use case
collection = chroma_client.get_or_create_collection(name="rag_demo")

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

all_chunks = []
all_metadatas = []  # track which document + which chunk index, for traceability

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
    metadatas=all_metadatas,   # Chroma stores this alongside each chunk
    ids=[f"chunk_{i}" for i in range(len(all_chunks))]
)

def rag_query_multi(query, collection, threshold=1.6, n_results=3):
    query_embedding = model.encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=n_results)

    for doc, dist, meta in zip(results["documents"][0], results["distances"][0], results["metadatas"][0]):
        print(f"[distance: {dist:.3f}] [source: {meta['source']}]\n{doc}\n")

    return results

# a query that COULD match either python or js docs
# rag_query_multi("How are key-value pairs stored?", collection_multi)
# rag_query_multi("Why is Python slow for CPU-heavy tasks?", collection_multi)
rag_query_multi("Which vector database should I use for production?", collection_multi)